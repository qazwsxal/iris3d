//! What an actor draws, as distinct from where it sits.
//!
//! An actor used to find its data by asking for its transform parent's
//! dataset, which welded three separate questions onto one edge: whose data do
//! I draw, whose transform do I inherit, and who owns my lifetime. Splitting
//! the first out into its own relationship is what lets two scene nodes draw a
//! single dataset — [`ActorOf`] and [`ChildOf`] may name different entities,
//! and then the data comes from one and the placement from the other.
//!
//! Both links are ordinary relationships and do not interact, so the common
//! case — drawing an object in place — is simply the two pointing at the same
//! entity.

use bevy::prelude::*;

use crate::counter::{GlobalIDCounter, UniqueID};

use super::{ActorKindId, Subset};

/// The object whose data this actor draws.
///
/// Deliberately not the transform parent. An actor of object `B` parented to
/// object `A` renders `B`'s arrays at `A`'s placement, which is how one
/// dataset appears in several places without being uploaded twice.
#[derive(Component, Debug, Clone, Copy)]
#[relationship(relationship_target = Actors)]
pub struct ActorOf(pub Entity);

/// The actors drawing an object, maintained by Bevy.
///
/// Note the absence of `linked_spawn`. It would be wrong here: an actor may be
/// sourced from one object while owned by another, so despawning a source
/// would silently destroy actors that belong elsewhere in the tree — and,
/// worse, do it without them appearing in the reply to whatever asked for the
/// deletion. Removal is explicit instead; see `scene::delete_object` and
/// [`reap_orphaned_actors`].
#[derive(Component, Debug)]
#[relationship_target(relationship = ActorOf)]
pub struct Actors(Vec<Entity>);

/// Spawns an actor of `source`, placed under `parent`.
///
/// The only way an actor is created, so that [`ActorOf`] is never absent on a
/// live one — [`reap_orphaned_actors`] treats its absence as proof the source
/// is gone.
///
/// `Transform` and `Visibility` make the entity a full participant in the
/// hierarchy, so it inherits its parent's placement and visibility. Bevy's
/// `Mesh3d` requires `Transform` but *not* `Visibility`, so a backend that only
/// added `Mesh3d` would leave an entity the visibility systems never collect.
///
/// `subset` is a parameter rather than something the caller folds into `extra`
/// because every backend queries one, so it cannot be optional — and a bundle
/// carrying the same component twice is a panic, not a last-write-wins.
pub fn spawn_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    source: Entity,
    parent: Entity,
    subset: Subset,
    extra: impl Bundle,
) -> (u64, Entity) {
    let id = counter.next();
    let entity = commands
        .spawn((
            ActorOf(source),
            ChildOf(parent),
            UniqueID(id),
            subset,
            Transform::default(),
            Visibility::default(),
            extra,
        ))
        .id();
    (id, entity)
}

/// Despawns actors whose source object is gone.
///
/// Bevy strips [`ActorOf`] when the source despawns, but leaves the entity
/// itself alone — and an actor that has already drawn something still holds a
/// `Mesh3d`, so what remains on screen is a render of data that no longer
/// exists. `scene::delete_object` removes them up front, so this is the safety
/// net for every other way a source can disappear.
pub fn reap_orphaned_actors(
    mut commands: Commands,
    orphans: Query<Entity, (With<ActorKindId>, Without<ActorOf>)>,
) {
    for entity in &orphans {
        commands.entity(entity).despawn();
        warn!("scene: despawned an actor whose source object was removed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_systems(Update, reap_orphaned_actors);
        app
    }

    fn actor(app: &mut App, source: Entity, parent: Entity) -> Entity {
        app.world_mut()
            .spawn((ActorKindId("surface"), ActorOf(source), ChildOf(parent)))
            .id()
    }

    /// An actor parented elsewhere outlives the despawn of its source, because
    /// nothing in the transform hierarchy connects the two. Bevy strips
    /// `ActorOf` and stops there, so without the reaper what is left is an
    /// entity still holding whatever mesh it last drew.
    #[test]
    fn reaps_an_actor_whose_source_is_gone() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let elsewhere = app.world_mut().spawn_empty().id();
        let actor = actor(&mut app, source, elsewhere);

        app.update();
        assert!(app.world().get_entity(actor).is_ok());

        app.world_mut().entity_mut(source).despawn();
        app.update();
        assert!(
            app.world().get_entity(actor).is_err(),
            "an actor of a deleted object must not survive"
        );
    }

    /// The ordinary case still goes through `Children`, so the reaper has
    /// nothing to do and must not be what removes it.
    #[test]
    fn leaves_actors_of_living_objects_alone() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let actor = actor(&mut app, source, source);

        app.update();
        app.update();
        assert!(app.world().get_entity(actor).is_ok());
    }

    /// Two links at two different entities is the arrangement the whole split
    /// exists for; check Bevy really does maintain both independently.
    #[test]
    fn source_and_parent_are_independent() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let elsewhere = app.world_mut().spawn_empty().id();
        let actor = actor(&mut app, source, elsewhere);
        app.update();

        let drawn_by: Vec<Entity> = app
            .world()
            .get::<Actors>(source)
            .expect("source lists what draws it")
            .iter()
            .collect();
        assert_eq!(drawn_by, vec![actor]);

        let children: Vec<Entity> = app
            .world()
            .get::<Children>(elsewhere)
            .expect("parent lists its children")
            .iter()
            .collect();
        assert_eq!(children, vec![actor]);

        assert!(app.world().get::<Actors>(elsewhere).is_none());
        assert!(app.world().get::<Children>(source).is_none());
    }
}
