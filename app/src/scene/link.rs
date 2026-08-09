//! What a representation draws, as distinct from where it sits.
//!
//! A representation used to find its data by asking for its transform parent's
//! dataset, which welded three separate questions onto one edge: whose data do
//! I draw, whose transform do I inherit, and who owns my lifetime. Splitting
//! the first out into its own relationship is what lets two scene nodes draw a
//! single dataset — [`RepresentationOf`] and [`ChildOf`] may name different
//! entities, and then the data comes from one and the placement from the other.
//!
//! Both links are ordinary relationships and do not interact, so the common
//! case — drawing an object in place — is simply the two pointing at the same
//! entity.

use bevy::prelude::*;

use crate::counter::{GlobalIDCounter, UniqueID};

use super::Representation;

/// The object whose data this representation draws.
///
/// Deliberately not the transform parent. A representation of object `B`
/// parented to object `A` renders `B`'s arrays at `A`'s placement, which is how
/// one dataset appears in several places without being uploaded twice.
#[derive(Component, Debug, Clone, Copy)]
#[relationship(relationship_target = Representations)]
pub struct RepresentationOf(pub Entity);

/// The representations drawing an object, maintained by Bevy.
///
/// Note the absence of `linked_spawn`. It would be wrong here: a representation
/// may be sourced from one object while owned by another, so despawning a
/// source would silently destroy representations that belong elsewhere in the
/// tree — and, worse, do it without them appearing in the reply to whatever
/// asked for the deletion. Removal is explicit instead; see
/// `scene::delete_object` and [`reap_orphaned_representations`].
#[derive(Component, Debug)]
#[relationship_target(relationship = RepresentationOf)]
pub struct Representations(Vec<Entity>);

/// Spawns a representation of `source`, placed under `parent`.
///
/// The only way a representation is created, so that [`RepresentationOf`] is
/// never absent on a live one — [`reap_orphaned_representations`] treats its
/// absence as proof the source is gone.
///
/// `Transform` and `Visibility` make the entity a full participant in the
/// hierarchy, so it inherits its parent's placement and visibility. Bevy's
/// `Mesh3d` requires `Transform` but *not* `Visibility`, so a backend that only
/// added `Mesh3d` would leave an entity the visibility systems never collect.
pub fn spawn_representation(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    source: Entity,
    parent: Entity,
    extra: impl Bundle,
) -> (u64, Entity) {
    let id = counter.next();
    let entity = commands
        .spawn((
            RepresentationOf(source),
            ChildOf(parent),
            UniqueID(id),
            Transform::default(),
            Visibility::default(),
            extra,
        ))
        .id();
    (id, entity)
}

/// Despawns representations whose source object is gone.
///
/// Bevy strips [`RepresentationOf`] when the source despawns, but leaves the
/// entity itself alone — and a representation that has already drawn something
/// still holds a `Mesh3d`, so what remains on screen is a render of data that no
/// longer exists. `scene::delete_object` removes them up front, so this is the
/// safety net for every other way a source can disappear.
pub fn reap_orphaned_representations(
    mut commands: Commands,
    orphans: Query<Entity, (With<Representation>, Without<RepresentationOf>)>,
) {
    for entity in &orphans {
        commands.entity(entity).despawn();
        warn!("scene: despawned a representation whose source object was removed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_systems(Update, reap_orphaned_representations);
        app
    }

    fn representation(app: &mut App, source: Entity, parent: Entity) -> Entity {
        app.world_mut()
            .spawn((
                Representation::Surface,
                RepresentationOf(source),
                ChildOf(parent),
            ))
            .id()
    }

    /// A representation parented elsewhere outlives the despawn of its source,
    /// because nothing in the transform hierarchy connects the two. Bevy strips
    /// `RepresentationOf` and stops there, so without the reaper what is left is
    /// an entity still holding whatever mesh it last drew.
    #[test]
    fn reaps_a_representation_whose_source_is_gone() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let elsewhere = app.world_mut().spawn_empty().id();
        let representation = representation(&mut app, source, elsewhere);

        app.update();
        assert!(app.world().get_entity(representation).is_ok());

        app.world_mut().entity_mut(source).despawn();
        app.update();
        assert!(
            app.world().get_entity(representation).is_err(),
            "a representation of a deleted object must not survive"
        );
    }

    /// The ordinary case still goes through `Children`, so the reaper has
    /// nothing to do and must not be what removes it.
    #[test]
    fn leaves_representations_of_living_objects_alone() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let representation = representation(&mut app, source, source);

        app.update();
        app.update();
        assert!(app.world().get_entity(representation).is_ok());
    }

    /// Two links at two different entities is the arrangement the whole split
    /// exists for; check Bevy really does maintain both independently.
    #[test]
    fn source_and_parent_are_independent() {
        let mut app = app();
        let source = app.world_mut().spawn_empty().id();
        let elsewhere = app.world_mut().spawn_empty().id();
        let representation = representation(&mut app, source, elsewhere);
        app.update();

        let drawn_by: Vec<Entity> = app
            .world()
            .get::<Representations>(source)
            .expect("source lists what draws it")
            .iter()
            .collect();
        assert_eq!(drawn_by, vec![representation]);

        let children: Vec<Entity> = app
            .world()
            .get::<Children>(elsewhere)
            .expect("parent lists its children")
            .iter()
            .collect();
        assert_eq!(children, vec![representation]);

        assert!(app.world().get::<Representations>(elsewhere).is_none());
        assert!(app.world().get::<Children>(source).is_none());
    }
}
