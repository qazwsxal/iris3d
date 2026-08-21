//! The object commands: creating, moving, reparenting, deleting and listing.
//!
//! An object is a place in the tree and a name. Everything here is about that
//! place — where it sits, what it hangs under, and what is drawn there.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use iris3d_core::counter::{GlobalIDCounter, UniqueID};
use iris3d_model::SceneError;

use super::actor_commands::summarise_actor;
use super::apply::{ActorQuery, Objects};
use super::link::Placement;
use super::{ActorSummary, Deleted, MAX_HIERARCHY_DEPTH, ObjectSummary, SceneObject};

/// Spawns an object entity and registers its handle. Returns the handle and a
/// summary of the new object.
///
/// The object is a place in the tree and a name. It holds no data and nothing
/// draws it: how to draw something is the caller's decision, and a client that
/// wants a particular representation asks `ListActorKinds` and names one.
pub(crate) fn spawn_object(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    index: &mut HashMap<u64, Entity>,
    object: SceneObject,
) -> (u64, ObjectSummary) {
    let id = counter.next_id();
    let name = object.name.clone();

    let spawned = commands
        .spawn((
            object,
            UniqueID(id),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    index.insert(id, spawned);

    let summary = ObjectSummary {
        id,
        name,
        // Nothing draws a new object. The caller adds an actor when it has
        // decided how it wants this drawn.
        actors: Vec::new(),
        parent: None,
    };

    info!("scene: added object {} \"{}\", not drawn", id, summary.name);

    (id, summary)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_parent(
    commands: &mut Commands,
    transforms: &mut Query<&mut Transform>,
    index: &HashMap<u64, Entity>,
    pending_parent: &mut HashMap<Entity, Option<Entity>>,
    child_of: &Query<&ChildOf>,
    globals: &Query<&GlobalTransform>,
    id: u64,
    parent: Option<u64>,
    keep_world_transform: bool,
) -> Result<(), SceneError> {
    let entity = *index.get(&id).ok_or(SceneError::NoSuchObject(id))?;

    let parent_entity = match parent {
        Some(parent_id) => Some(
            *index
                .get(&parent_id)
                .ok_or(SceneError::NoSuchObject(parent_id))?,
        ),
        None => None,
    };

    if let (Some(target), Some(parent_id)) = (parent_entity, parent)
        && would_cycle(entity, target, pending_parent, child_of)
    {
        return Err(SceneError::WouldCycle {
            object: id,
            parent: parent_id,
        });
    }

    if keep_world_transform {
        // Both world transforms have to be real ones. An object created
        // earlier in this same tick has neither: it is in `index` because the
        // drain recorded it, but `Commands` has not spawned it yet, so its
        // `GlobalTransform` has not been propagated and its `Transform` is not
        // there to write to. Reading through those misses would place the
        // object as though it and its new parent both sat at the origin —
        // wrong, and silently so, which is worse than the refusal the three
        // sibling commands already give for the same situation.
        let world = *globals
            .get(entity)
            .map_err(|_| SceneError::NoSuchObject(id))?;
        let parent_world = match (parent_entity, parent) {
            (Some(target), Some(parent_id)) => *globals
                .get(target)
                .map_err(|_| SceneError::NoSuchObject(parent_id))?,
            // Detaching: the world is the parent, and it is at the origin.
            _ => GlobalTransform::default(),
        };
        let Ok(mut local) = transforms.get_mut(entity) else {
            return Err(SceneError::NoSuchObject(id));
        };
        *local = world.reparented_to(&parent_world);
    }

    match parent_entity {
        Some(target) => {
            commands.entity(entity).insert(ChildOf(target));
            info!("scene: object {id} parented to {}", parent.unwrap());
        }
        None => {
            commands.entity(entity).remove::<ChildOf>();
            info!("scene: object {id} detached to root");
        }
    }
    pending_parent.insert(entity, parent_entity);

    Ok(())
}

/// True if parenting `object` to `parent` would close a loop.
pub(crate) fn would_cycle(
    object: Entity,
    parent: Entity,
    pending_parent: &HashMap<Entity, Option<Entity>>,
    child_of: &Query<&ChildOf>,
) -> bool {
    let mut cursor = Some(parent);
    for _ in 0..MAX_HIERARCHY_DEPTH {
        let Some(current) = cursor else { return false };
        if current == object {
            return true;
        }
        // Parent changes made earlier this tick are not visible to the query.
        cursor = match pending_parent.get(&current) {
            Some(pending) => *pending,
            None => child_of.get(current).ok().map(|link| link.parent()),
        };
    }
    // Ran out of depth: treat a malformed hierarchy as a cycle rather than
    // letting transform propagation panic on it.
    true
}

pub(crate) fn effective_parent(
    entity: Entity,
    pending_parent: &HashMap<Entity, Option<Entity>>,
    child_of: &Query<&ChildOf>,
) -> Option<Entity> {
    match pending_parent.get(&entity) {
        Some(pending) => *pending,
        None => child_of.get(entity).ok().map(|link| link.parent()),
    }
}

/// Removes one object, returning the handles actually removed.
///
/// Deletes exactly what was named. Child *objects* are detached first and
/// become roots, because a transform is a place and they still have one with
/// nothing above them.
///
/// The other children are placements, and those go. A placement is one copy of
/// an actor, under this object; the actor itself is not in the tree and is
/// untouched, so what a deletion costs it is that one appearance. If it was the
/// only one the actor draws nowhere, which needs nothing arranging — there is
/// simply no placement left. `sync_placements` drops the dead parent from its
/// list, and `RemoveActor` is what destroys an actor outright.
///
/// An actor and its placements are separate precisely so that this is true: one
/// drawing shown in three places must not be destroyed by tidying up one of
/// them.
pub(crate) fn delete_object(
    commands: &mut Commands,
    index: &mut HashMap<u64, Entity>,
    pending_parent: &mut HashMap<Entity, Option<Entity>>,
    objects: &Objects,
    children: &Query<&Children>,
    id: u64,
) -> Deleted {
    let Some(entity) = index.get(&id).copied() else {
        return Deleted::default();
    };

    if let Ok(list) = children.get(entity) {
        for child in list.iter().filter(|child| objects.contains(*child)) {
            commands.entity(child).remove::<ChildOf>();
            // Queued, so anything listing the scene later in this same drain
            // would otherwise report a parent about to be despawned.
            pending_parent.insert(child, None);
        }
    }

    // The handle is what the lookup above matched on, so there is nothing to
    // read back off the entity.
    index.remove(&id);
    commands.entity(entity).despawn();
    let deleted = Deleted { objects: vec![id] };

    info!("scene: deleted object {id}; its child objects are roots now");

    deleted
}

/// Sets an object's local placement. Unset components are left alone.
pub(crate) fn set_transform(
    transforms: &mut Query<&mut Transform>,
    index: &HashMap<u64, Entity>,
    id: u64,
    translation: Option<Vec3>,
    rotation: Option<Quat>,
    scale: Option<Vec3>,
) -> Result<(), SceneError> {
    let entity = index
        .get(&id)
        .copied()
        .ok_or(SceneError::NoSuchObject(id))?;
    let mut local = transforms
        .get_mut(entity)
        .map_err(|_| SceneError::NoSuchObject(id))?;
    if let Some(translation) = translation {
        local.translation = translation;
    }
    if let Some(rotation) = rotation {
        local.rotation = rotation;
    }
    if let Some(scale) = scale {
        local.scale = scale;
    }
    Ok(())
}

/// Every object, with what is drawn under each.
///
/// What is drawn is the placements among an object's children, each standing for
/// an actor; the rest of its children are nested objects. An actor under several
/// objects is reported under each of them, which is the truth — it is one actor,
/// appearing in several places.
#[allow(clippy::too_many_arguments)]
pub(crate) fn list_objects(
    objects: &Objects,
    children: &Query<&Children>,
    placements: &Query<&Placement>,
    actors: &ActorQuery,
    ids: &Query<&UniqueID>,
    child_of: &Query<&ChildOf>,
    pending_parent: &HashMap<Entity, Option<Entity>>,
) -> Vec<ObjectSummary> {
    let mut listing: Vec<ObjectSummary> = objects
        .iter()
        .map(|(entity, id, object)| {
            let drawn = children
                .get(entity)
                .into_iter()
                .flat_map(|list| list.iter())
                .filter_map(|child| {
                    let actor = placements.get(child).ok()?.0;
                    summarise_actor(actors.get(actor).ok()?, ids)
                })
                .collect();
            let parent = effective_parent(entity, pending_parent, child_of)
                .and_then(|p| ids.get(p).ok())
                .map(|unique| unique.0);
            summarise(id.0, object, drawn, parent)
        })
        .collect();
    listing.sort_by_key(|summary| summary.id);
    listing
}

pub(crate) fn summarise(
    id: u64,
    object: &SceneObject,
    actors: Vec<ActorSummary>,
    parent: Option<u64>,
) -> ObjectSummary {
    ObjectSummary {
        id,
        name: object.name.clone(),
        actors,
        parent,
    }
}
