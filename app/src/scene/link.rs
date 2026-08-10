//! Where an actor sits.
//!
//! There used to be a second link, `ActorOf`, naming the object whose *data* an
//! actor drew — separate from `ChildOf`, which named where it was placed. The
//! split existed so two scene nodes could draw a single dataset: point the two
//! links at different objects, take the data from one and the placement from
//! the other.
//!
//! Binding does that better. An actor names the arrays it reads, so drawing one
//! dataset in two places is two actors binding the same array under two
//! parents — and it works per array rather than per object, which the old link
//! could not express at all. What was left of `ActorOf` was a lifetime and
//! grouping edge wearing a name that said data.
//!
//! So an actor is now simply a child of the object it is drawn under, and
//! Bevy's own hierarchy owns its lifetime.

use bevy::prelude::*;

use crate::counter::{GlobalIDCounter, UniqueID};

use super::Subset;

/// Spawns an actor under `parent`.
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
    parent: Entity,
    subset: Subset,
    extra: impl Bundle,
) -> (u64, Entity) {
    let id = counter.next();
    let entity = commands
        .spawn((
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::registry::ActorKindId;

    /// Despawning an object takes the actors drawn under it. This is what
    /// replaced `reap_orphaned_actors`: an actor used to be linked to a source
    /// object it need not be a child of, so the hierarchy could not be trusted
    /// to clean up and a system swept for actors whose source had vanished.
    /// An actor is a plain child now, and Bevy's recursive despawn is the whole
    /// of its lifetime.
    #[test]
    fn an_actor_dies_with_the_object_it_is_drawn_under() {
        let mut app = App::new();
        let mut counter = GlobalIDCounter::default();

        let object = app.world_mut().spawn_empty().id();
        let mut commands = app.world_mut().commands();
        let (_, actor) = spawn_actor(
            &mut commands,
            &mut counter,
            object,
            Subset::All,
            ActorKindId("surface"),
        );
        app.world_mut().flush();
        assert!(app.world().get_entity(actor).is_ok());

        app.world_mut().entity_mut(object).despawn();
        assert!(
            app.world().get_entity(actor).is_err(),
            "an actor must not outlive the node it is drawn under"
        );
    }
}
