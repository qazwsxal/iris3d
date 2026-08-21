//! Where an actor is drawn, which may be several places at once.
//!
//! An actor and its placements are separate entities, and have to be:
//! `ChildOf` holds one parent, so an actor that *was* its placement could only
//! ever appear in one place.
//!
//! The actor entity holds the definition — its kind, its
//! parameters, its bindings — and owns whatever drawable output its kind builds
//! for it. It is not in the tree and it never renders. A [`Placement`]
//! entity per parent is a child of an object and carries *clones of those
//! handles*, which is what puts one drawing in several places for the cost of
//! one mesh.
//!
//! Sharing the handles rather than the geometry is the whole point. A kind
//! rebuilds into the asset it already owns, so one rebuild updates every
//! placement, and editing an actor edits every copy of it at once. Two actors
//! binding the same array can do neither: they are two definitions, changed one
//! at a time.
//!
//! Note that there is no link naming the object whose *data* an actor draws.
//! There is nothing for one to say: an actor binds arrays directly, so it has
//! any number of sources and, through placements, any number of places.

use bevy::platform::collections::HashSet;
use bevy::prelude::*;

use crate::counter::{GlobalIDCounter, UniqueID};

use super::SceneObject;

/// The objects an actor is drawn under.
///
/// A set in meaning if not in type: an actor is either drawn under an object or
/// it is not, and a repeat would stack two identical meshes in one place.
/// Empty is allowed and draws nothing — the state deleting the last object an
/// actor was under leaves it in.
#[derive(Component, Debug, Default, Clone, PartialEq, Eq)]
pub struct Parents(pub Vec<Entity>);

/// One appearance of an actor, under one object.
///
/// A child of that object, so it inherits the transform and visibility, and it
/// carries copies of the actor's mesh and material handles. Nothing else: no
/// parameters, no bindings, no handle of its own. It is not addressable from
/// outside either — an object is what a client places and an actor is what it
/// configures, which leaves a placement nothing to own. Two copies somewhere
/// different is two objects.
#[derive(Component, Debug, Clone, Copy)]
#[relationship(relationship_target = Placements)]
pub struct Placement(pub Entity);

/// An actor's placements, maintained by Bevy.
///
/// `linked_spawn`, unlike the `Actors` list this replaced: a placement is a
/// copy of one actor and means nothing without it, so removing an actor takes
/// its placements with it. The other direction is Bevy's ordinary recursive
/// despawn — a placement is a child of its object, so deleting the object
/// takes it, and [`sync_placements`] then drops the dead parent.
#[derive(Component, Debug, Default, Deref)]
#[relationship_target(relationship = Placement, linked_spawn)]
pub struct Placements(Vec<Entity>);

/// Whether the client asked for this actor to be drawn.
///
/// A flag of its own rather than the actor's `Visibility`, because that one is
/// already spoken for: the actor holds the mesh it owns, and its `Visibility`
/// is pinned to `Hidden` to keep that mesh off screen. See [`spawn_actor`].
/// This is the setting, and [`apply_shown`] gives it to the placements, which
/// are what actually draw.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shown(pub bool);

impl Default for Shown {
    fn default() -> Self {
        Self(true)
    }
}

/// Spawns an actor to be drawn under `parents`.
///
/// `Visibility::Hidden`, always and permanently. An actor owns the mesh its
/// kind builds for it, and a mesh on a visible entity is drawn — the actor is a
/// root, so it would appear at the origin as a second copy of everything.
///
/// Leaving `Visibility` off does not work, however much it looks like it
/// should. `Mesh3d`'s derive requires only `Transform`, but `VisibilityPlugin`
/// adds `Mesh3d -> Visibility` as a *runtime* required component, so the moment
/// a kind inserts a mesh Bevy supplies a default `Visibility::Inherited` and
/// the actor renders. A required component is only filled in when absent, so
/// carrying `Hidden` from the start is what holds.
///
/// Hiding the actor costs its placements nothing: they are children of the
/// objects, not of the actor, so nothing inherits this. Their own visibility
/// comes from [`Shown`] via [`apply_shown`].
///
pub fn spawn_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    parents: Vec<Entity>,
    extra: impl Bundle,
) -> (u64, Entity) {
    let id = counter.next_id();
    let entity = commands
        .spawn((
            UniqueID(id),
            Parents(parents),
            Shown(true),
            Visibility::Hidden,
            extra,
        ))
        .id();
    (id, entity)
}

/// Makes each actor's placements match the parents it names.
///
/// Every frame rather than only on change, because a parent can also stop
/// existing: deleting an object despawns the placements under it, and this is
/// what notices and drops the entry so nothing tries to rebuild them.
///
/// Adds and removes rather than rebuilding the set. A placement's mesh handle
/// is written by whichever kind owns the actor, so respawning them each
/// frame would throw that away and flicker.
pub fn sync_placements(
    mut commands: Commands,
    mut actors: Query<(Entity, &mut Parents, Option<&Placements>)>,
    parent_of: Query<&ChildOf, With<Placement>>,
    objects: Query<(), With<SceneObject>>,
) {
    for (actor, mut parents, placements) in &mut actors {
        // An object that has been deleted is not a parent any more. Asked of
        // the objects query rather than of entity liveness, so an entity that
        // is alive but not an object is dropped just as surely.
        if parents.0.iter().any(|parent| !objects.contains(*parent)) {
            parents.0.retain(|parent| objects.contains(*parent));
        }

        let wanted: HashSet<Entity> = parents.0.iter().copied().collect();
        let mut held: HashSet<Entity> = HashSet::new();
        for placement in placements.into_iter().flat_map(|list| list.iter()) {
            // A placement whose object was despawned is already gone; what is
            // left here is one under an object the actor no longer names.
            match parent_of.get(placement).map(|link| link.parent()) {
                Ok(parent) if wanted.contains(&parent) => {
                    held.insert(parent);
                }
                _ => commands.entity(placement).despawn(),
            }
        }
        for parent in wanted.difference(&held) {
            // `Transform` explicitly, not as a side effect of something a
            // backend adds later. A placement is a *place* — that is the whole
            // of what it is — so it carries one whatever ends up drawn there.
            //
            // Relying on `Mesh3d` to require it is not enough: a backend that
            // puts its instances *under* the placement rather than on it would
            // leave the placement with no `GlobalTransform` for them to
            // inherit, and Bevy warns once per instance.
            commands.spawn((
                Placement(actor),
                ChildOf(*parent),
                Transform::default(),
                Visibility::default(),
            ));
        }
    }
}

/// Hides or shows every placement of an actor together.
///
/// The setting belongs to the actor — a client asked for *this drawing* to be
/// hidden, not for one copy of it — so it is applied to all of them.
pub fn apply_shown(actors: Query<&Shown>, mut placements: Query<(&Placement, &mut Visibility)>) {
    for (placement, mut visibility) in &mut placements {
        let wanted = match actors.get(placement.0) {
            Ok(Shown(true)) => Visibility::Inherited,
            // An actor mid-despawn takes its placements with it; hiding them
            // first costs nothing and avoids a frame of orphaned geometry.
            _ => Visibility::Hidden,
        };
        if *visibility != wanted {
            *visibility = wanted;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::registry::ActorKindId;

    fn app() -> App {
        let mut app = App::new();
        // The one thing `VisibilityPlugin` does that matters here, without
        // dragging in the rest of it. Leave this out and an actor looks safely
        // invisible in a test while rendering in the app.
        app.register_required_components::<Mesh3d, Visibility>();
        app.add_systems(Update, (sync_placements, apply_shown).chain());
        app
    }

    fn object(app: &mut App) -> Entity {
        app.world_mut()
            .spawn((
                SceneObject { name: "o".into() },
                Transform::default(),
                Visibility::default(),
            ))
            .id()
    }

    fn actor(app: &mut App, parents: Vec<Entity>) -> Entity {
        let mut counter = GlobalIDCounter::default();
        let mut commands = app.world_mut().commands();
        let (_, entity) = spawn_actor(&mut commands, &mut counter, parents, ActorKindId("surface"));
        app.world_mut().flush();
        entity
    }

    /// Which objects an actor currently appears under.
    fn placed_under(app: &mut App, actor: Entity) -> Vec<Entity> {
        let mut found: Vec<Entity> = app
            .world_mut()
            .query::<(&Placement, &ChildOf)>()
            .iter(app.world())
            .filter(|(placement, _)| placement.0 == actor)
            .map(|(_, link)| link.parent())
            .collect();
        found.sort();
        found
    }

    /// The actor itself must never be drawn, even once it holds a mesh.
    ///
    /// It is a root, so a visible one puts a copy of everything at the origin
    /// alongside the real placements — which is exactly what happened when this
    /// relied on the actor simply having no `Visibility`. `VisibilityPlugin`
    /// registers `Mesh3d -> Visibility` at runtime, so the first kind to draw
    /// handed it a default `Inherited`.
    #[test]
    fn an_actor_stays_hidden_once_a_kind_gives_it_a_mesh() {
        let mut app = app();
        let object = object(&mut app);
        let actor = actor(&mut app, vec![object]);
        app.update();

        app.world_mut()
            .entity_mut(actor)
            .insert(Mesh3d(Handle::default()));
        app.update();

        assert_eq!(
            app.world().get::<Visibility>(actor),
            Some(&Visibility::Hidden),
            "a mesh on a visible actor is a second copy of everything at the origin"
        );
        // And the setting a client sees is untouched by that.
        assert_eq!(app.world().get::<Shown>(actor), Some(&Shown(true)));
    }

    /// Hiding the actor must not hide what it draws. The placements are
    /// children of the objects, so nothing inherits it.
    #[test]
    fn hiding_the_actor_does_not_hide_its_placements() {
        let mut app = app();
        let object = object(&mut app);
        let actor = actor(&mut app, vec![object]);
        app.update();

        let placement = app
            .world_mut()
            .query::<(Entity, &Placement)>()
            .iter(app.world())
            .find(|(_, p)| p.0 == actor)
            .map(|(entity, _)| entity)
            .expect("a placement");
        assert_eq!(
            app.world().get::<Visibility>(placement),
            Some(&Visibility::Inherited),
            "the copy under the object is what draws"
        );
    }

    /// One actor, two objects, two placements. The point of the split.
    #[test]
    fn an_actor_is_placed_under_every_parent_it_names() {
        let mut app = app();
        let (first, second) = (object(&mut app), object(&mut app));
        let actor = actor(&mut app, vec![first, second]);
        app.update();

        let mut wanted = vec![first, second];
        wanted.sort();
        assert_eq!(placed_under(&mut app, actor), wanted);
    }

    /// Dropping a parent costs that placement and leaves the others alone.
    #[test]
    fn dropping_a_parent_removes_only_that_placement() {
        let mut app = app();
        let (kept, dropped) = (object(&mut app), object(&mut app));
        let actor = actor(&mut app, vec![kept, dropped]);
        app.update();

        app.world_mut()
            .entity_mut(actor)
            .insert(Parents(vec![kept]));
        app.update();

        assert_eq!(placed_under(&mut app, actor), vec![kept]);
    }

    /// Losing an object costs the actor that placement and nothing else.
    ///
    /// This is what replaced parking a parentless actor under a hidden node:
    /// with the mesh on the placements, an actor with none simply draws
    /// nowhere, and needs nothing to arrange that.
    #[test]
    fn deleting_an_object_costs_the_actor_only_that_placement() {
        let mut app = app();
        let (kept, doomed) = (object(&mut app), object(&mut app));
        let actor = actor(&mut app, vec![kept, doomed]);
        app.update();

        app.world_mut().entity_mut(doomed).despawn();
        app.update();

        assert_eq!(placed_under(&mut app, actor), vec![kept]);
        assert!(
            app.world().get_entity(actor).is_ok(),
            "the actor outlives the object it was drawn under"
        );
        assert_eq!(
            app.world().get::<Parents>(actor),
            Some(&Parents(vec![kept])),
            "and stops claiming a parent that no longer exists"
        );
    }

    /// An actor under nothing is not drawn, and needs no special state to
    /// arrange it: there is no placement, so there is nothing on screen.
    #[test]
    fn an_actor_under_no_object_has_no_placements() {
        let mut app = app();
        let object = object(&mut app);
        let actor = actor(&mut app, vec![object]);
        app.update();

        app.world_mut().entity_mut(actor).insert(Parents(vec![]));
        app.update();

        assert!(placed_under(&mut app, actor).is_empty());
        assert!(app.world().get_entity(actor).is_ok());
    }

    /// Removing an actor takes every copy of it.
    #[test]
    fn removing_an_actor_takes_its_placements() {
        let mut app = app();
        let (first, second) = (object(&mut app), object(&mut app));
        let actor = actor(&mut app, vec![first, second]);
        app.update();

        app.world_mut().entity_mut(actor).despawn();
        app.update();

        assert_eq!(
            app.world_mut()
                .query::<&Placement>()
                .iter(app.world())
                .len(),
            0,
            "a placement is a copy of an actor and means nothing without it"
        );
    }
}
