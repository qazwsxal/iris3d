//! Draining the command queue and applying it to the world.
//!
//! One system, [`apply_scene_commands`], with a handler per command. The match
//! is the router and every arm is a named function, so the shape of what the
//! scene can be asked is readable in one screen and the work is beside the
//! command that asks for it.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use iris3d_core::counter::{GlobalIDCounter, UniqueID};
use iris3d_core::redraw::KeepAwake;
use iris3d_data::{DataArray, DataStore};

use super::actor_commands::{add_actor, list_actor_kinds, list_actors, remove_actor, set_actor};
use super::data_commands::{list_data, release_data, upload_data};
use super::link::{Parents, Placement, Shown};
use super::object_commands::{
    delete_object, list_objects, set_parent, set_transform, spawn_object,
};
use super::registry::{ActorKindId, ActorParams, ActorRegistry};
use super::{
    CommandBus, SceneCommand, SceneObject,
};

/// Read-only view of the objects in the scene.
///
/// An object's children are placements and nested objects, told apart by what
/// each entity carries. An actor is not among them — a placement is a copy of
/// one, and the actor itself sits outside the tree.
pub(crate) type Objects<'w, 's> = Query<'w, 's, (Entity, &'static UniqueID, &'static SceneObject)>;

/// Mutable view of the actor entities.
///
/// One query rather than several, because a read-only query over the same
/// components as a `&mut` one is a conflict Bevy rejects at schedule init even
/// when the two could never match the same entity.
pub(crate) type ActorQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static ActorKindId,
        &'static mut ActorParams,
        &'static mut Shown,
        &'static mut Parents,
    ),
    With<ActorKindId>,
>;

/// Everything a client's data lives in, as one system parameter.
///
/// Bundled because [`apply_scene_commands`] sits near Bevy's ceiling of sixteen
/// system parameters, and because the two are always wanted together: the store
/// says which handle names what, and the asset collection holds it.
///
/// No `Assets<Mesh>`: a mesh handle only ever reaches the store through a
/// filter's geometry output, which the filter graph allocates.
#[derive(bevy::ecs::system::SystemParam)]
pub struct HeldData<'w> {
    pub arrays: ResMut<'w, Assets<DataArray>>,
    pub store: ResMut<'w, DataStore>,
}

/// What [`ActorQuery`] yields when read rather than written.
pub(crate) type ActorItem<'a> = (
    Entity,
    &'a UniqueID,
    &'a ActorKindId,
    &'a ActorParams,
    &'a Shown,
    &'a Parents,
);

/// Drains commands submitted from outside the ECS and applies them to the
/// world. Replies are best-effort: a caller that has hung up is not an error.
///
/// Public so rendering backends can order themselves after it and pick up new
/// actors on the frame they appear.
///
/// Structural changes go through Bevy's deferred `Commands`, so the queries here
/// still show the pre-tick hierarchy. Parent changes made earlier in this same
/// drain are therefore tracked in `pending_parent` and consulted during cycle
/// validation — without that, two `SetParent` commands arriving in one tick
/// could each look safe in isolation and together form a cycle.
#[allow(clippy::too_many_arguments)]
pub fn apply_scene_commands(
    mut commands: Commands,
    bus: Res<CommandBus>,
    mut counter: ResMut<GlobalIDCounter>,
    registry: Res<ActorRegistry>,
    held: HeldData,
    mut transforms: Query<&mut Transform>,
    objects: Objects,
    ids: Query<&UniqueID>,
    children: Query<&Children>,
    child_of: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    mut actors: ActorQuery,
    placements: Query<&Placement>,
    mut awake: ResMut<KeepAwake>,
) {
    let batch: Vec<SceneCommand> = std::iter::from_fn(|| bus.try_recv().ok()).collect();
    if batch.is_empty() {
        return;
    }
    // Unpacked straight back into the two names the body uses. They are one
    // system parameter only because Bevy's ceiling is sixteen of them and this
    // system is near it — the grouping is a packing detail, not a concept.
    let HeldData {
        mut arrays,
        mut store,
    } = held;

    // What these commands ask for takes several frames to appear, and the
    // update loop is otherwise asleep. Hold it open until the scene has caught
    // up with them.
    awake.nudge();

    let mut index: HashMap<u64, Entity> = objects.iter().map(|(e, id, ..)| (id.0, e)).collect();
    // Actors get the same treatment as objects: spawned entities are recorded
    // on the spot, so two commands arriving in one tick — add an actor, then
    // configure it — can see each other despite the queries still showing the
    // pre-tick world.
    let mut drawn: HashMap<u64, Entity> = actors
        .iter()
        .map(|(entity, id, ..)| (id.0, entity))
        .collect();
    let mut pending_parent: HashMap<Entity, Option<Entity>> = HashMap::new();

    for command in batch {
        match command {
            SceneCommand::UploadData {
                arrays: uploaded,
                reply,
            } => {
                let _ = reply.send(upload_data(&mut counter, &mut arrays, &mut store, uploaded));
            }

            SceneCommand::ListData { reply } => {
                let _ = reply.send(list_data(&store));
            }

            SceneCommand::ReleaseData { ids, reply } => {
                let _ = reply.send(release_data(&mut store, ids));
            }

            SceneCommand::CreateObject { name, reply } => {
                let object = SceneObject { name };
                let (_, summary) = spawn_object(&mut commands, &mut counter, &mut index, object);
                let _ = reply.send(summary);
            }

            SceneCommand::SetParent {
                id,
                parent,
                keep_world_transform,
                reply,
            } => {
                let _ = reply.send(set_parent(
                    &mut commands,
                    &mut transforms,
                    &index,
                    &mut pending_parent,
                    &child_of,
                    &globals,
                    id,
                    parent,
                    keep_world_transform,
                ));
            }

            SceneCommand::SetTransform {
                id,
                translation,
                rotation,
                scale,
                reply,
            } => {
                let _ = reply.send(set_transform(
                    &mut transforms,
                    &index,
                    id,
                    translation,
                    rotation,
                    scale,
                ));
            }

            SceneCommand::ListObjects { reply } => {
                let _ = reply.send(list_objects(
                    &objects,
                    &children,
                    &placements,
                    &actors,
                    &ids,
                    &child_of,
                    &pending_parent,
                ));
            }

            SceneCommand::DeleteObject { id, reply } => {
                let _ = reply.send(delete_object(
                    &mut commands,
                    &mut index,
                    &mut pending_parent,
                    &objects,
                    &children,
                    id,
                ));
            }

            SceneCommand::AddActor {
                kind,
                parents,
                params,
                reply,
            } => {
                let _ = reply.send(add_actor(
                    &mut commands,
                    &mut counter,
                    &registry,
                    &mut index,
                    &mut drawn,
                    parents,
                    kind,
                    params,
                    &store,
                ));
            }

            SceneCommand::SetActor {
                id,
                params,
                visible,
                parents,
                reply,
            } => {
                let _ = reply.send(set_actor(
                    &registry,
                    &drawn,
                    &index,
                    &mut actors,
                    &ids,
                    &store,
                    id,
                    params,
                    visible,
                    parents,
                ));
            }

            SceneCommand::RemoveActor { id, reply } => {
                let _ = reply.send(remove_actor(&mut commands, &mut drawn, id));
            }

            SceneCommand::ListActors { parent, reply } => {
                let _ = reply.send(list_actors(&actors, &ids, &index, parent));
            }

            SceneCommand::ListActorKinds { reply } => {
                let _ = reply.send(list_actor_kinds(&registry));
            }
        }
    }
}
