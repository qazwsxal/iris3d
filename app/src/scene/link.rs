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
//! So an actor is now simply a child of the object it is drawn under. That is
//! placement and grouping only — not lifetime. Deleting an object detaches its
//! actors rather than destroying them, because an actor is defined by the
//! arrays it binds and those outlive every node in the tree. `RemoveActor` is
//! what ends an actor, and `SetActor`'s `parent` is what places one again.

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

    /// An actor spawns as a child, so it inherits the object's placement and
    /// visibility. That is all the link is for — `delete_object` detaches
    /// rather than cascading, so the hierarchy does not own an actor's
    /// lifetime; see `an_actor_outlives_the_object_it_was_drawn_under`.
    #[test]
    fn an_actor_is_spawned_as_a_child_of_its_object() {
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

        assert_eq!(
            app.world().get::<ChildOf>(actor).map(|link| link.parent()),
            Some(object),
            "an actor has to be a child, or it inherits no transform"
        );
        assert!(
            app.world().get::<Visibility>(actor).is_some(),
            "without Visibility the visibility systems never collect it"
        );
    }
}
