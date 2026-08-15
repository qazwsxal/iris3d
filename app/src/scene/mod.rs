//! The scene: objects held in the Bevy world, and the command interface used
//! to get data into and out of them.
//!
//! Three layers, deliberately separated:
//!
//! - [`data`] — raw arrays, held flat in a [`DataStore`] and referred to by
//!   handle. They belong to no object.
//! - [`actor`] — how something gets *drawn*, as its own entity, binding the
//!   arrays it reads.
//! - [`link`] — where an actor sits, which is not the same question as what it
//!   reads.
//!
//! An object is a place in the tree and nothing else. It used to hold data as
//! well, and a dataset component saying what shape that data made, which meant
//! "put these numbers in the scene" and "put a node in the tree" were one
//! operation. Splitting them is what lets one array feed several actors, and one
//! actor read arrays that arrived separately.
//!
//! Nothing here knows about gRPC, and nothing here draws. A rendering backend
//! plugs in by consuming actors and their bindings — see [`crate::draw`], which
//! holds the pathways and the part they share. Which one is running is a launch
//! choice; this module is the same either way, and that is the point of the
//! split.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};
use tokio::sync::oneshot;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::filter::{self, FilterKindSummary, FilterSummary};
use crate::grpc::GrpcBridge;
use crate::redraw::KeepAwake;

pub mod data;
pub mod link;
pub mod registry;
pub mod subset;

// Only what other modules reach for. The rest stays available under its own
// module path — this is a binary crate, so unused re-exports are just noise.
pub use data::{BufferMeta, DataArray, DataStore, Dtype, HeldMeta, NamedBuffer};
pub use link::{Parents, Placement, Shown};
pub use registry::{ActorKindId, ActorParams, ActorRegistry};
pub use subset::{Subset, SubsetEncoding};

/// Ceiling on how far the ancestor walk will climb before giving up. Guards
/// against a pre-existing malformed hierarchy sending validation into a loop.
const MAX_HIERARCHY_DEPTH: usize = 4096;

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<DataArray>()
            .init_resource::<DataStore>()
            .add_systems(
                Update,
                // Order matters: the drain creates actors, style components and
                // bindings are derived from the parameters it wrote, and the
                // placements follow the parents it set.
                (
                    apply_scene_commands,
                    registry::apply_actor_params,
                    link::sync_placements,
                    link::apply_shown,
                )
                    .chain(),
            );
    }
}

/// A place in the scene tree, and a name for it.
///
/// It used to hold data too — the arrays it was uploaded with, a dataset
/// component saying what shape they made, and a `Fields` map saying what they
/// meant. An actor binds the arrays it draws, so none of that was read any
/// more, and what an object *is* has collapsed into where it is.
///
/// Paired with a [`UniqueID`] carrying its handle, a transform, and whatever
/// actors and child objects hang under it.
#[derive(Component, Debug)]
pub struct SceneObject {
    pub name: String,
}

/// One held handle, described without its contents.
#[derive(Debug, Clone)]
pub struct DataSummary {
    pub id: u64,
    pub meta: HeldMeta,
}

/// A description of an object in the scene.
///
/// No buffers and no byte count: an object holds no data to describe. What is
/// resident is a question about the arrays, and `ListData` answers it.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub id: u64,
    pub name: String,
    /// Everything drawing here.
    pub actors: Vec<ActorSummary>,
    /// Parent in the scene tree, `None` for a root.
    pub parent: Option<u64>,
}

/// A description of one way something is being drawn.
#[derive(Debug, Clone)]
pub struct ActorSummary {
    pub id: u64,
    /// Registered kind id — see [`ActorRegistry`].
    pub kind: String,
    /// Every object it is drawn under, in handle order.
    ///
    /// Any number, including none. One actor under several objects is one
    /// drawing appearing in several places — changed once, changed everywhere
    /// — and an empty list draws nothing, which is where deleting the last
    /// object it was under leaves it.
    pub parents: Vec<u64>,
    pub params: registry::ParamMap,
    pub visible: bool,
    /// How much of the bound data is drawn, or `None` for all of it. The values
    /// are not carried back — the caller sent them, and they can be large.
    pub subset: Option<SubsetSummary>,
}

/// An actor's selection, described without returning it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsetSummary {
    pub encoding: SubsetEncoding,
    pub association: data::Association,
    /// How many elements it keeps.
    pub selected: u64,
}

/// A way of drawing that a backend has registered, described for a client.
#[derive(Debug, Clone)]
pub struct KindSummary {
    pub id: String,
    pub label: String,
    pub params: &'static [registry::ParamSpec],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    NoSuchObject(u64),
    NoSuchActor(u64),
    /// A handle that names no held array. Distinct from `NoSuchObject` because
    /// the three handle spaces share one sequence — passing an object handle
    /// where an array was wanted is a plausible mistake worth naming precisely.
    NoSuchData(u64),
    /// An input the kind cannot draw without was left unbound.
    MissingInput {
        kind: String,
        input: &'static str,
    },
    /// The bound array is the wrong element type or shape for the input.
    BadBinding {
        kind: String,
        input: &'static str,
        /// What is wrong, in the words of the input's own declaration.
        reason: String,
    },
    /// The running backend registered no kind by that name, so nothing could
    /// draw it.
    ///
    /// Carries the pathway as well as the name, because a kind can be perfectly
    /// real under another backend and simply absent here. Saying which one is
    /// running is the difference between "you mistyped it" and "you are on the
    /// wrong pathway".
    UnknownKind {
        kind: String,
        backend: &'static str,
    },
    NoSuchFilter(u64),
    /// No filter kind by that name. No backend to name, unlike
    /// [`UnknownKind`](Self::UnknownKind): a filter derives data and has no
    /// pipeline, so the same set exists in every build.
    UnknownFilterKind {
        kind: String,
    },
    /// Binding this would put a filter's own output somewhere in its own
    /// inputs.
    ///
    /// Refused rather than tolerated. A cycle is not a slow render: each run
    /// rewrites an array that marks the next one stale, so the graph never comes
    /// to rest and the app never sleeps.
    FilterCycle {
        filter: u64,
        input: &'static str,
    },
    /// A handle a filter is writing cannot be forgotten on its own.
    ///
    /// Releasing it would leave the filter producing into nothing, which looks
    /// exactly like a filter that has stopped working. Remove the filter.
    StillGenerated {
        data: u64,
        filter: u64,
    },
    /// The requested parent is the object itself or one of its descendants.
    ///
    /// Rejecting this is not optional: Bevy's transform propagation *panics* on
    /// a hierarchy cycle, so allowing one would let a client crash the
    /// application with two calls.
    WouldCycle {
        object: u64,
        parent: u64,
    },
}

impl Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::NoSuchObject(id) => write!(f, "no object with handle {id}"),
            SceneError::NoSuchActor(id) => {
                write!(f, "no actor with handle {id}")
            }
            SceneError::NoSuchData(id) => write!(f, "no uploaded array with handle {id}"),
            SceneError::MissingInput { kind, input } => write!(
                f,
                "actor kind \"{kind}\" cannot draw without an array bound to \"{input}\""
            ),
            SceneError::BadBinding {
                kind,
                input,
                reason,
            } => write!(
                f,
                "the array bound to \"{input}\" of actor kind \"{kind}\" {reason}"
            ),
            SceneError::UnknownKind { kind, backend } => write!(
                f,
                "the {backend} backend has no actor kind \"{kind}\" — ask \
                 ListActorKinds for the ones this pathway draws"
            ),
            SceneError::WouldCycle { object, parent } => write!(
                f,
                "object {object} cannot be parented to {parent}: {parent} is {object} \
                 or one of its descendants"
            ),
            SceneError::NoSuchFilter(id) => write!(f, "no filter with handle {id}"),
            SceneError::UnknownFilterKind { kind } => write!(
                f,
                "there is no filter kind \"{kind}\" — ask ListFilterKinds for the \
                 ones this build has"
            ),
            SceneError::FilterCycle { filter, input } => write!(
                f,
                "binding that to \"{input}\" would feed filter {filter} its own \
                 output, directly or through others"
            ),
            SceneError::StillGenerated { data, filter } => write!(
                f,
                "array {data} is an output of filter {filter} and cannot be released \
                 on its own — remove the filter instead"
            ),
        }
    }
}

impl std::error::Error for SceneError {}

/// Work submitted to the scene from outside the ECS.
///
/// Each variant carries the channel its result should be written to, so the
/// submitting task can await a reply without the scene knowing who asked.
/// Buffers are fully assembled before they arrive here — the ECS tick that
/// applies a command never does the transfer.
#[derive(Debug)]
pub enum SceneCommand {
    /// Takes in arrays on their own. No object, no actor — the reply hands back
    /// a handle per array, and something else decides what they are for.
    UploadData {
        arrays: Vec<NamedBuffer>,
        reply: oneshot::Sender<Vec<DataSummary>>,
    },
    ListData {
        reply: oneshot::Sender<Vec<DataSummary>>,
    },
    /// Forgets arrays. Reports the handles that were held, so a caller learns
    /// which of them it was wrong about.
    ReleaseData {
        ids: Vec<u64>,
        reply: oneshot::Sender<Vec<u64>>,
    },
    /// Creates an object holding no data, for use as a grouping node.
    CreateObject {
        name: String,
        reply: oneshot::Sender<ObjectSummary>,
    },
    SetParent {
        id: u64,
        /// `None` detaches, making the object a root.
        parent: Option<u64>,
        keep_world_transform: bool,
        reply: oneshot::Sender<Result<(), SceneError>>,
    },
    /// Sets an object's local placement. Unset components are left alone.
    SetTransform {
        id: u64,
        translation: Option<Vec3>,
        rotation: Option<Quat>,
        scale: Option<Vec3>,
        reply: oneshot::Sender<Result<(), SceneError>>,
    },
    ListObjects {
        reply: oneshot::Sender<Vec<ObjectSummary>>,
    },
    DeleteObject {
        id: u64,
        /// Deletes exactly this node. Every child is detached and becomes a
        /// root, actors as much as objects.
        reply: oneshot::Sender<Deleted>,
    },

    /// Draws something under an object. Adds; never replaces.
    AddActor {
        /// The objects to draw under, whose transforms it inherits. Empty
        /// makes one, since an actor has no place of its own. *What* it draws
        /// is in `params`, as bindings.
        parents: Vec<u64>,
        /// Which registered kind draws it. Named by the caller, always: the
        /// server has no opinion on how a dataset should look.
        kind: String,
        /// Partial. Anything unset takes the kind's **default**, there being no
        /// previous value.
        params: registry::ParamMap,
        /// `None` draws the whole dataset.
        subset: Option<subset::SubsetRequest>,
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    SetActor {
        id: u64,
        /// Partial. Anything unset keeps its **current** value — the opposite
        /// of [`AddActor`](Self::AddActor), because here there is one.
        params: registry::ParamMap,
        visible: Option<bool>,
        /// `None` leaves the selection alone; `Some(None)` clears it back to
        /// drawing everything. Absent and cleared have to be distinguishable,
        /// which is what the nesting buys.
        subset: Option<Option<subset::SubsetRequest>>,
        /// Replace the set of objects it is drawn under. `None` leaves them
        /// alone; `Some(vec![])` takes it off screen without removing it.
        parents: Option<Vec<u64>>,
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    RemoveActor {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    ListActors {
        /// Restrict to those drawing one object.
        parent: Option<u64>,
        reply: oneshot::Sender<Result<Vec<ActorSummary>, SceneError>>,
    },
    ListActorKinds {
        reply: oneshot::Sender<Vec<KindSummary>>,
    },

    /// Derives arrays from arrays. Draws nothing.
    ///
    /// The reply carries a handle for each declared output, allocated by this
    /// call, so the next thing a caller does can bind one — there is no waiting
    /// for a first run to learn what they are.
    AddFilter {
        /// Which registered kind. Named by the caller, as an actor's is.
        kind: String,
        /// Partial. Anything unset takes the kind's **default**.
        params: registry::ParamMap,
        reply: oneshot::Sender<Result<FilterSummary, SceneError>>,
    },
    SetFilter {
        id: u64,
        /// Partial. Anything unset keeps its **current** value — the opposite
        /// of [`AddFilter`](Self::AddFilter), because here there is one.
        params: registry::ParamMap,
        reply: oneshot::Sender<Result<FilterSummary, SceneError>>,
    },
    /// Removes a filter and forgets the arrays it was writing.
    RemoveFilter {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    ListFilters {
        reply: oneshot::Sender<Vec<FilterSummary>>,
    },
    ListFilterKinds {
        reply: oneshot::Sender<Vec<FilterKindSummary>>,
    },
}

/// What a deletion took with it.
///
/// Only ever the object named. Actors used to be listed here too, because a
/// deletion destroyed the ones drawn under it; they are detached now, so
/// nothing else goes.
#[derive(Debug, Default, Clone)]
pub struct Deleted {
    pub objects: Vec<u64>,
}

/// Read-only view of the objects in the scene.
///
/// An object's children are placements and nested objects, told apart by what
/// each entity carries. An actor is no longer among them — a placement is a
/// copy of one, and the actor itself sits outside the tree.
type Objects<'w, 's> = Query<'w, 's, (Entity, &'static UniqueID, &'static SceneObject)>;

/// Mutable view of the actor entities.
///
/// One query rather than several, because a read-only query over the same
/// components as a `&mut` one is a conflict Bevy rejects at schedule init even
/// when the two could never match the same entity.
type ActorQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static ActorKindId,
        &'static mut ActorParams,
        &'static mut Shown,
        &'static mut Subset,
        &'static mut Parents,
    ),
    With<ActorKindId>,
>;

/// Everything a client's data lives in, as one system parameter.
///
/// Bundled because [`apply_scene_commands`] was already at Bevy's ceiling of
/// sixteen system parameters, and adding `Assets<Mesh>` for geometry outputs
/// pushed it over. Grouped rather than split up because these three are always
/// wanted together: the store says which handle names what, and the two asset
/// collections hold it.
#[derive(bevy::ecs::system::SystemParam)]
pub struct HeldData<'w> {
    pub arrays: ResMut<'w, Assets<DataArray>>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub store: ResMut<'w, DataStore>,
}

/// What [`ActorQuery`] yields when read rather than written.
type ActorItem<'a> = (
    Entity,
    &'a UniqueID,
    &'a ActorKindId,
    &'a ActorParams,
    &'a Shown,
    &'a Subset,
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
    bridge: Res<GrpcBridge>,
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
    mut filters: crate::filter::Filters,
) {
    let batch: Vec<SceneCommand> = std::iter::from_fn(|| bridge.try_recv().ok()).collect();
    if batch.is_empty() {
        return;
    }
    // Unpacked straight back into the three names the body uses. They are one
    // system parameter only because Bevy's ceiling is sixteen of them and this
    // system is at it — the grouping is a packing detail, not a concept.
    let HeldData {
        mut arrays,
        mut meshes,
        mut store,
    } = held;

    // Who reads whose output, built once and kept current as commands are
    // applied. Two commands arriving in one tick have to see each other, or
    // adding a filter and then pointing another at it would look like two
    // unrelated calls and the cycle check would miss the pair.
    let mut graph = crate::filter::Graph::build(&filters.registry, &filters.entities);

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
            // Arrays with no object around them. The bytes become assets exactly
            // as an object's would; the only difference is who holds the handle.
            SceneCommand::UploadData {
                arrays: uploaded,
                reply,
            } => {
                let summaries: Vec<(DataSummary, String)> = uploaded
                    .into_iter()
                    .map(|buffer| {
                        let id = counter.next();
                        let meta = buffer.meta;
                        let handle = arrays.add(DataArray {
                            dtype: meta.dtype,
                            shape: meta.shape.clone(),
                            data: buffer.data,
                            strings: buffer.strings,
                        });
                        store.insert(id, meta.clone(), handle);
                        let name = meta.name.clone();
                        (
                            DataSummary {
                                id,
                                meta: HeldMeta::Array(meta),
                            },
                            name,
                        )
                    })
                    .collect::<Vec<_>>();
                info!(
                    "scene: took in {} array(s): {}",
                    summaries.len(),
                    summaries
                        .iter()
                        .map(|(array, name)| format!("{}={name}", array.id))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let _ = reply.send(summaries.into_iter().map(|(summary, _)| summary).collect());
            }

            // Arrays first, then meshes, each in handle order. Both are things
            // a client holds and can bind, so both belong in one listing rather
            // than making geometry a call of its own.
            SceneCommand::ListData { reply } => {
                let listing = store
                    .iter()
                    .map(|(id, array)| DataSummary {
                        id,
                        meta: HeldMeta::Array(array.meta.clone()),
                    })
                    .chain(store.iter_geometry().map(|(id, mesh)| DataSummary {
                        id,
                        meta: HeldMeta::Geometry(mesh.meta.clone()),
                    }))
                    .collect();
                let _ = reply.send(listing);
            }

            SceneCommand::ReleaseData { ids, reply } => {
                let released: Vec<u64> = ids
                    .into_iter()
                    .filter(|id| {
                        // An array a filter writes is not the caller's to
                        // forget: releasing it would leave the filter producing
                        // into nothing, which looks exactly like a filter that
                        // has broken. Reported rather than refused, because the
                        // call takes a batch and one bad handle should not lose
                        // the others.
                        match graph.producer_of(*id) {
                            Some(filter) => {
                                warn!("{}", SceneError::StillGenerated { data: *id, filter });
                                false
                            }
                            None => store.remove(*id),
                        }
                    })
                    .collect();
                let _ = reply.send(released);
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
                let result = set_parent(
                    &mut commands,
                    &mut transforms,
                    &index,
                    &mut pending_parent,
                    &child_of,
                    &globals,
                    id,
                    parent,
                    keep_world_transform,
                );
                let _ = reply.send(result);
            }

            SceneCommand::SetTransform {
                id,
                translation,
                rotation,
                scale,
                reply,
            } => {
                let result = index
                    .get(&id)
                    .copied()
                    .ok_or(SceneError::NoSuchObject(id))
                    .and_then(|entity| {
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
                    });
                let _ = reply.send(result);
            }

            SceneCommand::ListObjects { reply } => {
                let mut listing: Vec<ObjectSummary> = objects
                    .iter()
                    .map(|(entity, id, object)| {
                        // What is drawn here are the placements among its
                        // children, each standing for an actor; the rest of its
                        // children are nested objects. An actor under several
                        // objects is reported under each of them, which is the
                        // truth — it is one actor, appearing in several places.
                        let drawn = children
                            .get(entity)
                            .into_iter()
                            .flat_map(|list| list.iter())
                            .filter_map(|child| {
                                let actor = placements.get(child).ok()?.0;
                                summarise_actor(actors.get(actor).ok()?, &ids, &arrays)
                            })
                            .collect();
                        let parent = effective_parent(entity, &pending_parent, &child_of)
                            .and_then(|p| ids.get(p).ok())
                            .map(|unique| unique.0);
                        summarise(id.0, object, drawn, parent)
                    })
                    .collect();
                listing.sort_by_key(|summary| summary.id);
                let _ = reply.send(listing);
            }

            SceneCommand::DeleteObject { id, reply } => {
                let removed = delete_object(
                    &mut commands,
                    &mut index,
                    &mut pending_parent,
                    &objects,
                    &children,
                    id,
                );
                let _ = reply.send(removed);
            }

            SceneCommand::AddActor {
                kind,
                parents,
                params,
                subset,
                reply,
            } => {
                let result = add_actor(
                    &mut commands,
                    &mut counter,
                    &registry,
                    &mut index,
                    &mut drawn,
                    parents,
                    kind,
                    params,
                    subset,
                    &mut arrays,
                    &store,
                );
                let _ = reply.send(result);
            }

            SceneCommand::SetActor {
                id,
                params,
                visible,
                subset,
                parents,
                reply,
            } => {
                let result = set_actor(
                    &registry,
                    &drawn,
                    &index,
                    &mut actors,
                    &ids,
                    &mut arrays,
                    &store,
                    id,
                    params,
                    visible,
                    subset,
                    parents,
                );
                let _ = reply.send(result);
            }

            SceneCommand::RemoveActor { id, reply } => {
                let existed = match drawn.remove(&id) {
                    Some(entity) => {
                        // Actors own nothing, so there is no subtree question
                        // of the kind object deletion has to answer.
                        commands.entity(entity).despawn();
                        info!("scene: removed actor {id}");
                        true
                    }
                    None => false,
                };
                let _ = reply.send(existed);
            }

            SceneCommand::ListActors { parent, reply } => {
                let filter = match parent {
                    Some(id) => match index.get(&id) {
                        Some(entity) => Some(*entity),
                        None => {
                            let _ = reply.send(Err(SceneError::NoSuchObject(id)));
                            continue;
                        }
                    },
                    None => None,
                };
                let mut listing: Vec<ActorSummary> = actors
                    .iter()
                    // Filtered on where an actor is drawn. It used to filter on
                    // whose data it read, which is no longer a thing an actor
                    // has — it reads arrays, and any number of them.
                    .filter(|item| filter.is_none_or(|object| item.6.0.contains(&object)))
                    .filter_map(|item| summarise_actor(item, &ids, &arrays))
                    .collect();
                listing.sort_by_key(|summary| summary.id);
                let _ = reply.send(Ok(listing));
            }

            SceneCommand::AddFilter {
                kind,
                params,
                reply,
            } => {
                let result = filter::add(
                    &mut commands,
                    &mut counter,
                    &filters.registry,
                    &mut graph,
                    &mut arrays,
                    &mut meshes,
                    &mut store,
                    kind,
                    params,
                );
                let _ = reply.send(result);
            }

            SceneCommand::SetFilter { id, params, reply } => {
                let result = filter::set(
                    &filters.registry,
                    &mut graph,
                    &store,
                    &mut filters.entities,
                    id,
                    params,
                );
                let _ = reply.send(result);
            }

            SceneCommand::RemoveFilter { id, reply } => {
                let removed =
                    filter::remove(&mut commands, &mut graph, &mut store, &filters.entities, id);
                let _ = reply.send(removed);
            }

            SceneCommand::ListFilters { reply } => {
                let _ = reply.send(filter::list(&filters.registry, &filters.entities));
            }

            SceneCommand::ListFilterKinds { reply } => {
                let _ = reply.send(filter::list_kinds(&filters.registry));
            }

            SceneCommand::ListActorKinds { reply } => {
                let kinds = registry
                    .iter()
                    .map(|kind| KindSummary {
                        id: kind.id.to_string(),
                        label: kind.label.to_string(),
                        params: kind.params,
                    })
                    .collect();
                let _ = reply.send(kinds);
            }
        }
    }
}

/// Checks that every input an actor kind reads is bound, and bound to something
/// it can actually read.
///
/// Separate from [`registry::ParamKind::sanitise`] on purpose. Sanitising judges
/// a value on its own and runs wherever a parameter is written; this needs the
/// [`DataStore`] to see what a handle points at, and the store is not reachable
/// from all of those places. So one answers "is this the right sort of value"
/// and the other "is that particular array or mesh the right shape".
///
/// Arrays and meshes share a handle space, so this is also where binding one
/// where the other belongs is caught and named.
fn check_bindings(
    kind: &registry::ActorKind,
    params: &registry::ParamMap,
    store: &DataStore,
) -> Result<(), SceneError> {
    for spec in kind.inputs() {
        let required = spec.kind.is_required();
        match registry::data(params, spec.id) {
            Some(id) => {
                let held = store.held(id).ok_or(SceneError::NoSuchData(id))?;
                spec.kind
                    .accepts(held)
                    .map_err(|reason| SceneError::BadBinding {
                        kind: kind.id.to_string(),
                        input: spec.id,
                        reason,
                    })?;
            }
            // An optional input left unbound is the normal case, not a fault.
            None if required => {
                return Err(SceneError::MissingInput {
                    kind: kind.id.to_string(),
                    input: spec.id,
                });
            }
            None => {}
        }
    }
    Ok(())
}

/// Adds a way of drawing, at any number of places in the tree.
///
/// `parents` may be empty, in which case an object is made to hold this actor.
/// An actor has no place of its own, so it needs at least one — and a caller
/// with nothing to group it under would otherwise open every single drawing
/// with the same `CreateObject` call.
///
/// Naming several draws one actor in several places. It stays one actor: one
/// mesh, one set of parameters, and every copy changes together.
#[allow(clippy::too_many_arguments)]
fn add_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &ActorRegistry,
    index: &mut HashMap<u64, Entity>,
    drawn: &mut HashMap<u64, Entity>,
    parents: Vec<u64>,
    kind: String,
    params: registry::ParamMap,
    subset: Option<subset::SubsetRequest>,
    arrays: &mut Assets<DataArray>,
    store: &DataStore,
) -> Result<ActorSummary, SceneError> {
    // Resolved before anything is created, so a bad handle builds nothing.
    let named: Vec<(u64, Entity)> = parents
        .into_iter()
        .map(|id| {
            index
                .get(&id)
                .copied()
                .map(|entity| (id, entity))
                .ok_or(SceneError::NoSuchObject(id))
        })
        .collect::<Result<_, _>>()?;

    // The caller names the kind. There is no default to fall back on: which
    // representation suits some data is a judgement, and the server has no
    // basis for it beyond the order the kinds happened to register in.
    //
    // Nor is there a check that the kind suits the object any more. It used to
    // ask `supports(DatasetKind)`, which only meant anything while an actor took
    // its data from the object it hung under. An actor binds its own arrays now,
    // so what the object holds — usually nothing — says nothing about what can
    // draw there. `check_bindings` is the real gate, and it asks about the data
    // rather than about the node.
    let registered = registry
        .get(&kind)
        .ok_or_else(|| SceneError::UnknownKind {
            kind: kind.clone(),
            backend: registry.backend(),
        })?;

    // Unset parameters take the kind's default: this is a new actor, so there
    // is no previous value to preserve.
    let params = registered.normalise(&params);
    check_bindings(registered, &params, store)?;

    let subset = subset.map_or(Subset::All, |request| request.into_subset(arrays));
    let summarised_subset = match &subset {
        Subset::All => None,
        Subset::Selected {
            encoding,
            association,
            ..
        } => subset::size(&subset, arrays).map(|selected| SubsetSummary {
            encoding: *encoding,
            association: *association,
            selected,
        }),
    };

    // Only now, with every reason to refuse behind us. Creating the object any
    // earlier would leave an empty one behind whenever an actor is rejected.
    let placed = match named.is_empty() {
        false => named,
        true => {
            let name = registered.label.to_string();
            let (id, _) = spawn_object(commands, counter, index, SceneObject { name });
            vec![(id, index[&id])]
        }
    };
    let (parents, entities): (Vec<u64>, Vec<Entity>) = placed.into_iter().unzip();

    let (id, entity) = link::spawn_actor(
        commands,
        counter,
        entities,
        subset,
        (
            ActorKindId(registered.id),
            ActorParams(params.clone()),
        ),
    );
    drawn.insert(id, entity);

    info!(
        "scene: actor {id} draws {} under object(s) {parents:?}",
        registered.id
    );

    Ok(ActorSummary {
        id,
        kind: registered.id.to_string(),
        parents,
        params,
        visible: true,
        subset: summarised_subset,
    })
}

/// Changes an existing actor, leaving anything unnamed alone.
#[allow(clippy::too_many_arguments)]
fn set_actor(
    registry: &ActorRegistry,
    drawn: &HashMap<u64, Entity>,
    index: &HashMap<u64, Entity>,
    actors: &mut ActorQuery,
    ids: &Query<&UniqueID>,
    arrays: &mut Assets<DataArray>,
    store: &DataStore,
    id: u64,
    params: registry::ParamMap,
    visible: Option<bool>,
    subset: Option<Option<subset::SubsetRequest>>,
    parents: Option<Vec<u64>>,
) -> Result<ActorSummary, SceneError> {
    let entity = *drawn.get(&id).ok_or(SceneError::NoSuchActor(id))?;

    // Resolved before anything is written, so a bad handle changes nothing.
    let moving_to = parents
        .map(|wanted| {
            wanted
                .into_iter()
                .map(|id| index.get(&id).copied().ok_or(SceneError::NoSuchObject(id)))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    let Ok(mut item) = actors.get_mut(entity) else {
        // In `drawn` but not yet in the world: added earlier in this same
        // drain, so the query has not seen it. Nothing is lost by asking again
        // next tick, and guessing would mean writing to an entity blind.
        return Err(SceneError::NoSuchActor(id));
    };

    let registered = registry
        .get(item.2.0)
        .ok_or_else(|| SceneError::UnknownKind {
            kind: item.2.0.to_string(),
            backend: registry.backend(),
        })?;

    // Merge rather than replace: a client changing one setting should not have
    // to restate the others, and omitting them must not silently reset them.
    // Merged into a copy, because the bindings are judged as a whole below and
    // a refusal has to leave the actor exactly as it was.
    let mut merged = item.3.0.clone();
    for (key, value) in params {
        let Some(value) = registered
            .spec(&key)
            .and_then(|spec| spec.kind.sanitise(value))
        else {
            warn!("scene: actor {id} has no parameter \"{key}\" of that type");
            continue;
        };
        // Clearing is a removal, not a value — see the same merge in
        // `filter::wire::set`. An input let go has to be absent from the map,
        // because absence is what every reader already understands as unbound.
        match value {
            registry::ParamValue::Unset => merged.remove(&key),
            value => merged.insert(key, value),
        };
    }

    // The same gate `add_actor` passes through. `sanitise` only says that a
    // handle is a handle; whether that particular array fits the input needs
    // the store, so without this a rebind to the wrong dtype or shape would be
    // refused when adding an actor and accepted when changing one.
    check_bindings(registered, &merged, store)?;
    // Written only when it differs, so a call that restates the parameters does
    // not mark them `Changed` and throw the geometry away to rebuild it
    // identically.
    item.3.set_if_neq(ActorParams(merged));

    if let Some(visible) = visible {
        *item.4 = Shown(visible);
    }
    if let Some(subset) = subset {
        *item.5 = subset.map_or(Subset::All, |request| request.into_subset(arrays));
    }

    if let Some(wanted) = moving_to {
        // No cycle check, unlike `set_parent`. Nothing is ever parented under
        // a placement, so no arrangement of them can close a loop.
        //
        // A plain write, not a queued command: `sync_placements` reads this
        // next and builds or drops placements to match, so adding a parent and
        // taking the last one away are the same operation.
        *item.6 = Parents(wanted);
    }

    summarise_actor(
        actors.get(entity).expect("the entity was just written"),
        ids,
        arrays,
    )
    .ok_or(SceneError::NoSuchActor(id))
}

/// Spawns an object entity, its default actor, and registers its handle.
/// Returns the handle and a summary of the new object.
#[allow(clippy::too_many_arguments)]
/// Adds an object to the world. It holds data and a place in the tree, and
/// nothing draws it.
///
/// Choosing how to draw something is not the server's decision to make. It
/// used to pick the first registered kind that supported the dataset, which
/// meant the server answered a question only the caller can — and answered it
/// out of whatever order the kinds happened to register in. A client that
/// wants the obvious representation asks `ListActorKinds` and names one.
fn spawn_object(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    index: &mut HashMap<u64, Entity>,
    object: SceneObject,
) -> (u64, ObjectSummary) {
    let id = counter.next();
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
fn set_parent(
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

    if let (Some(target), Some(parent_id)) = (parent_entity, parent) {
        if would_cycle(entity, target, pending_parent, child_of) {
            return Err(SceneError::WouldCycle {
                object: id,
                parent: parent_id,
            });
        }
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
fn would_cycle(
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

fn effective_parent(
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
/// Actors used to die here, back when an actor *was* its placement. That is the
/// whole reason the two are separate: one drawing shown in three places should
/// not be destroyed by tidying up one of them.
fn delete_object(
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

/// Describes one actor from its query item.
///
/// `Option` only so callers can filter with `?`; an actor always describes.
fn summarise_actor(
    (_, id, kind, params, shown, subset, parents): ActorItem<'_>,
    ids: &Query<&UniqueID>,
    arrays: &Assets<DataArray>,
) -> Option<ActorSummary> {
    // Read straight off the component, unlike an object's parent. Placements
    // are built from this by a later system rather than by a queued command, so
    // there is no deferred write for a listing in this same drain to miss.
    let mut parents: Vec<u64> = parents
        .0
        .iter()
        .filter_map(|parent| ids.get(*parent).ok())
        .map(|unique| unique.0)
        .collect();
    parents.sort_unstable();

    Some(ActorSummary {
        id: id.0,
        kind: kind.0.to_string(),
        parents,
        params: params.0.clone(),
        // What this actor was told. An object hidden above one of its
        // placements still hides that copy, which is `InheritedVisibility`'s
        // business and not reported here.
        visible: shown.0,
        subset: match subset {
            Subset::All => None,
            Subset::Selected {
                encoding,
                association,
                ..
            } => subset::size(subset, arrays).map(|selected| SubsetSummary {
                encoding: *encoding,
                association: *association,
                selected,
            }),
        },
    })
}

fn summarise(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::GlobalIDCounter;
    use crate::grpc::GrpcBridge;
    use crate::redraw::KeepAwake;

    fn app() -> App {
        let mut app = App::new();
        // Enough of the world for the drain to run: the assets it ingests into,
        // the handle counter, an empty registry (nothing here asks to be drawn)
        // and the channel it reads. `TransformPlugin` is what makes
        // `GlobalTransform` mean anything, which the reparent path depends on.
        app.add_plugins(TransformPlugin);
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        // Geometry is an asset like any other, and `mark_dirty` watches it: a
        // filter rewriting a mesh has to reach the actors drawing it.
        app.add_message::<AssetEvent<Mesh>>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<DataStore>();
        app.init_resource::<GlobalIDCounter>();
        app.init_resource::<ActorRegistry>();
        // The filter half of what the drain reads. Only the registry and the
        // entities: nothing here runs a filter, because these tests are about
        // the commands rather than the work.
        app.init_resource::<filter::FilterRegistry>();
        // No pathway is added here, so nothing would otherwise name one, and the
        // messages that quote the backend would read "the no backend has...".
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .served_by("test");
        app.init_resource::<KeepAwake>();
        app.init_resource::<GrpcBridge>();
        // Placements follow the parents the drain writes, so the whole chain
        // has to run for a listing to reflect what is on screen.
        app.add_systems(
            Update,
            (
                apply_scene_commands,
                link::sync_placements,
                link::apply_shown,
            )
                .chain(),
        );
        app
    }

    /// Queues a command the way the gRPC side does, and hands back the reply.
    fn send<T>(
        app: &App,
        make: impl FnOnce(oneshot::Sender<T>) -> SceneCommand,
    ) -> oneshot::Receiver<T> {
        let (tx, rx) = oneshot::channel();
        app.world()
            .resource::<GrpcBridge>()
            .sender()
            .send(make(tx))
            .expect("the scene is draining");
        rx
    }

    fn create(app: &App, name: &str) -> oneshot::Receiver<ObjectSummary> {
        send(app, |reply| SceneCommand::CreateObject {
            name: name.into(),
            reply,
        })
    }

    fn place(app: &App, id: u64, x: f32) -> oneshot::Receiver<Result<(), SceneError>> {
        send(app, |reply| SceneCommand::SetTransform {
            id,
            translation: Some(Vec3::new(x, 0.0, 0.0)),
            rotation: None,
            scale: None,
            reply,
        })
    }

    fn reparent(
        app: &App,
        id: u64,
        parent: Option<u64>,
    ) -> oneshot::Receiver<Result<(), SceneError>> {
        send(app, |reply| SceneCommand::SetParent {
            id,
            parent,
            keep_world_transform: true,
            reply,
        })
    }

    /// A kind that draws nothing and needs nothing, so a test can make actors
    /// without a rendering backend compiled in.
    fn marker(app: &mut App) {
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(registry::ActorKind {
                id: "marker",
                label: "Marker",
                params: &[],
                apply: |_, _| {},
            });
    }

    fn two_objects(app: &mut App) -> (u64, u64) {
        let mut first = create(app, "first");
        let mut second = create(app, "second");
        app.update();
        (
            first.try_recv().expect("a reply").id,
            second.try_recv().expect("a reply").id,
        )
    }

    fn add(app: &mut App, parents: Vec<u64>) -> u64 {
        let mut added = send(app, |reply| SceneCommand::AddActor {
            parents,
            kind: "marker".into(),
            params: registry::ParamMap::default(),
            subset: None,
            reply,
        });
        app.update();
        added.try_recv().expect("a reply").expect("an actor").id
    }

    /// How many copies of anything are on screen.
    fn placements(app: &mut App) -> usize {
        app.world_mut()
            .query::<&Placement>()
            .iter(app.world())
            .len()
    }

    fn array(name: &str, bytes: usize) -> NamedBuffer {
        NamedBuffer {
            meta: BufferMeta {
                name: name.into(),
                dtype: Dtype::Uint8,
                shape: vec![bytes as u64],
            },
            data: vec![0; bytes],
            strings: Vec::new(),
        }
    }

    /// The same, of a shape the caller chooses, for tests about what an input
    /// will accept.
    fn shaped(name: &str, shape: &[u64]) -> NamedBuffer {
        let elements: u64 = shape.iter().product();
        NamedBuffer {
            meta: BufferMeta {
                name: name.into(),
                dtype: Dtype::Uint8,
                shape: shape.to_vec(),
            },
            data: vec![0; elements as usize],
            strings: Vec::new(),
        }
    }

    /// Arrays arrive on their own: a handle each, no object, no actor. Data used
    /// to be reachable only by making an object out of it, which conflated
    /// "hold these numbers" with "put a node in the tree".
    #[test]
    fn uploaded_arrays_are_held_without_an_object() {
        let mut app = app();
        let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
            arrays: vec![array("xyz", 12), array("t", 4)],
            reply,
        });
        app.update();

        let summaries = uploaded.try_recv().expect("a reply");
        let handles: Vec<u64> = summaries.iter().map(|array| array.id).collect();
        assert_eq!(summaries.len(), 2);
        assert_eq!(
            summaries
                .iter()
                .map(|a| a.meta.name())
                .collect::<Vec<_>>(),
            ["xyz", "t"],
            "handles come back in declaration order"
        );
        assert_eq!(
            app.world().resource::<DataStore>().iter().count(),
            2,
            "the store keeps the bytes alive"
        );
        let objects = app
            .world_mut()
            .query::<&SceneObject>()
            .iter(app.world())
            .count();
        assert_eq!(objects, 0, "an upload of arrays creates no object");

        // Forgetting reports what was actually held, so a caller learns which
        // of its handles it was wrong about.
        let mut released = send(&app, |reply| SceneCommand::ReleaseData {
            ids: vec![handles[0], 9_999],
            reply,
        });
        app.update();
        assert_eq!(released.try_recv().expect("a reply"), vec![handles[0]]);
        assert_eq!(app.world().resource::<DataStore>().iter().count(), 1);
    }

    /// The ordinary case: both objects are really in the world, so the child's
    /// local transform absorbs the parent's and it does not move.
    #[test]
    fn keeping_the_world_transform_offsets_the_child() {
        let mut app = app();
        let mut parent = create(&app, "parent");
        let mut child = create(&app, "child");
        app.update();
        let parent = parent.try_recv().expect("a reply").id;
        let child = child.try_recv().expect("a reply").id;

        place(&app, parent, 5.0);
        place(&app, child, 1.0);
        app.update();

        let mut moved = reparent(&app, child, Some(parent));
        app.update();
        assert_eq!(moved.try_recv().expect("a reply"), Ok(()));

        let entity = app
            .world_mut()
            .query::<(Entity, &UniqueID)>()
            .iter(app.world())
            .find(|(_, id)| id.0 == child)
            .map(|(entity, _)| entity)
            .expect("the child is in the world");
        let local = app.world().get::<Transform>(entity).expect("a transform");
        assert_eq!(
            local.translation,
            Vec3::new(-4.0, 0.0, 0.0),
            "the child should stay where it was in world space"
        );
    }

    /// The same command against an object created earlier in the *same* drain.
    /// Its handle is already in `index`, but `Commands` has not spawned it, so
    /// neither world transform exists. Answering with the origin would misplace
    /// the object without saying so, so the command has to fail instead.
    ///
    /// No client can reach this today — naming the object needs the handle, and
    /// the handle only arrives in a reply, by which point the spawn has landed.
    /// The test builds the batch by hand, since handles are allocated in order.
    #[test]
    fn refuses_to_keep_the_world_transform_of_an_object_made_this_tick() {
        let mut app = app();
        let mut parent = create(&app, "parent");
        app.update();
        let parent = parent.try_recv().expect("a reply").id;

        // Both in one batch, the second naming what the first will create.
        let mut child = create(&app, "child");
        let mut moved = reparent(&app, parent + 1, Some(parent));
        app.update();

        assert_eq!(
            child.try_recv().expect("a reply").id,
            parent + 1,
            "handles come out of one sequence, so the batch could name it"
        );
        assert_eq!(
            moved.try_recv().expect("a reply"),
            Err(SceneError::NoSuchObject(parent + 1)),
            "a silent misplacement is worse than a refusal"
        );
    }

    /// An actor with no parent named gets an object made for it, named after
    /// its kind.
    ///
    /// An actor has no place of its own, so it has to end up under something.
    /// Refusing instead would make `CreateObject` the opening line of every
    /// drawing a client does, to make a node it has no other use for.
    #[test]
    fn an_actor_with_no_parent_is_given_an_object() {
        let mut app = app();
        marker(&mut app);

        let mut added = send(&app, |reply| SceneCommand::AddActor {
            parents: vec![],
            kind: "marker".into(),
            params: registry::ParamMap::default(),
            subset: None,
            reply,
        });
        app.update();
        let actor = added.try_recv().expect("a reply").expect("an actor");

        let mut listed = send(&app, |reply| SceneCommand::ListObjects { reply });
        app.update();
        let objects = listed.try_recv().expect("a reply");
        assert_eq!(
            objects
                .iter()
                .map(|o| (o.id, o.name.as_str()))
                .collect::<Vec<_>>(),
            vec![(actor.parents[0], "Marker")],
            "exactly one object, named after the kind, holding the actor"
        );
    }

    /// A refused actor leaves no object behind.
    ///
    /// The object is created last, after every check. Creating it up front
    /// would litter the scene with empty nodes whenever a client got a binding
    /// or a kind name wrong.
    #[test]
    fn an_actor_that_cannot_be_added_creates_nothing() {
        let mut app = app();

        let mut added = send(&app, |reply| SceneCommand::AddActor {
            parents: vec![],
            kind: "no-such-kind".into(),
            params: registry::ParamMap::default(),
            subset: None,
            reply,
        });
        app.update();
        let refusal = added.try_recv().expect("a reply").err();
        assert_eq!(
            refusal,
            Some(SceneError::UnknownKind {
                kind: "no-such-kind".into(),
                backend: "test",
            })
        );
        // The pathway belongs in the message, not only in the variant. A kind
        // can be perfectly real under another backend, and a client that cannot
        // tell "you mistyped it" from "you are on the wrong pathway" has to
        // guess.
        let said = refusal.expect("a refusal").to_string();
        assert!(said.contains("test backend"), "{said}");

        let mut listed = send(&app, |reply| SceneCommand::ListObjects { reply });
        app.update();
        assert!(
            listed.try_recv().expect("a reply").is_empty(),
            "a refusal must not leave an empty object behind"
        );
    }

    /// A rebind to an array the input cannot read is refused, and the actor
    /// keeps the array it had.
    ///
    /// `SetActor` used to judge each value with `sanitise` alone, which for an
    /// array input only confirms that a handle is a handle. An input's declared
    /// element type and shape therefore held when the actor was added and
    /// stopped holding the moment it was changed — which is the call a client
    /// uses to rebind geometry.
    #[test]
    fn an_actor_keeps_its_binding_when_a_rebind_is_refused() {
        const INPUTS: &[registry::ParamSpec] = &[registry::ParamSpec {
            id: "field",
            label: "Field",
            kind: registry::ParamKind::Array {
                dtypes: &[Dtype::Uint8],
                shape: &[0],
                required: true,
                structural: true,
            },
        }];

        let mut app = app();
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(registry::ActorKind {
                id: "bound",
                label: "Bound",
                params: INPUTS,
                apply: |_, _| {},
            });

        let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
            arrays: vec![
                array("first", 4),
                shaped("wrong", &[2, 3]),
                array("second", 8),
            ],
            reply,
        });
        app.update();
        let handles: Vec<u64> = uploaded
            .try_recv()
            .expect("a reply")
            .iter()
            .map(|array| array.id)
            .collect();
        let (first, wrong, second) = (handles[0], handles[1], handles[2]);

        let bind = |handle: u64| {
            let mut params = registry::ParamMap::default();
            params.insert("field".into(), registry::ParamValue::Data(handle));
            params
        };

        let mut added = send(&app, |reply| SceneCommand::AddActor {
            parents: vec![],
            kind: "bound".into(),
            params: bind(first),
            subset: None,
            reply,
        });
        app.update();
        let actor = added.try_recv().expect("a reply").expect("an actor").id;

        let mut refused = send(&app, |reply| SceneCommand::SetActor {
            id: actor,
            params: bind(wrong),
            visible: None,
            subset: None,
            parents: None,
            reply,
        });
        app.update();
        let refusal = refused
            .try_recv()
            .expect("a reply")
            .expect_err("a [2, 3] array does not fit an [n] input");
        let SceneError::BadBinding {
            ref kind,
            input,
            ref reason,
        } = refusal
        else {
            panic!("expected a refused binding, got {refusal:?}");
        };
        assert_eq!((kind.as_str(), input), ("bound", "field"));
        // The reason reaches the client that got it wrong, as it does from
        // `AddActor`.
        assert!(reason.contains("[2, 3]"), "{reason}");
        assert!(reason.contains("[n]"), "{reason}");

        let bound = |app: &mut App| {
            let mut listed = send(app, |reply| SceneCommand::ListActors {
                parent: None,
                reply,
            });
            app.update();
            let listing = listed.try_recv().expect("a reply").expect("a listing");
            registry::data(&listing[0].params, "field")
        };
        assert_eq!(
            bound(&mut app),
            Some(first),
            "a refused rebind must leave the actor exactly as it was"
        );

        // And a binding that does fit still goes through: the check refuses
        // what the input cannot read, not every change.
        let mut accepted = send(&app, |reply| SceneCommand::SetActor {
            id: actor,
            params: bind(second),
            visible: None,
            subset: None,
            parents: None,
            reply,
        });
        app.update();
        accepted.try_recv().expect("a reply").expect("an actor");
        assert_eq!(bound(&mut app), Some(second));
    }

    /// An optional input can be let go of, and a required one cannot.
    ///
    /// The merge rule makes this need a value of its own: a partial map leaves
    /// anything absent alone, so absence already means "unchanged" and cannot
    /// also mean "clear". Without `Unset` an optional input could be bound once
    /// and never released, by any client.
    #[test]
    fn an_optional_input_can_be_unbound_and_a_required_one_cannot() {
        const INPUTS: &[registry::ParamSpec] = &[
            registry::ParamSpec {
                id: "field",
                label: "Field",
                kind: registry::ParamKind::Array {
                    dtypes: &[Dtype::Uint8],
                    shape: &[0],
                    required: true,
                    structural: true,
                },
            },
            registry::ParamSpec {
                id: "mask",
                label: "Mask",
                kind: registry::ParamKind::Array {
                    dtypes: &[Dtype::Uint8],
                    shape: &[0],
                    required: false,
                    structural: true,
                },
            },
        ];

        let mut app = app();
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(registry::ActorKind {
                id: "maskable",
                label: "Maskable",
                params: INPUTS,
                apply: |_, _| {},
            });

        let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
            arrays: vec![array("field", 4), array("mask", 4)],
            reply,
        });
        app.update();
        let handles: Vec<u64> = uploaded
            .try_recv()
            .expect("a reply")
            .iter()
            .map(|array| array.id)
            .collect();

        let mut params = registry::ParamMap::default();
        params.insert("field".into(), registry::ParamValue::Data(handles[0]));
        params.insert("mask".into(), registry::ParamValue::Data(handles[1]));
        let mut added = send(&app, |reply| SceneCommand::AddActor {
            parents: vec![],
            kind: "maskable".into(),
            params,
            subset: None,
            reply,
        });
        app.update();
        let actor = added.try_recv().expect("a reply").expect("an actor").id;

        let clear = |input: &str| {
            let mut params = registry::ParamMap::default();
            params.insert(input.into(), registry::ParamValue::Unset);
            params
        };
        let bound = |app: &mut App, input: &'static str| {
            let mut listed = send(app, |reply| SceneCommand::ListActors {
                parent: None,
                reply,
            });
            app.update();
            let listing = listed.try_recv().expect("a reply").expect("a listing");
            registry::data(&listing[0].params, input)
        };

        let mut cleared = send(&app, |reply| SceneCommand::SetActor {
            id: actor,
            params: clear("mask"),
            visible: None,
            subset: None,
            parents: None,
            reply,
        });
        app.update();
        cleared.try_recv().expect("a reply").expect("an actor");
        assert_eq!(bound(&mut app, "mask"), None, "the mask should be let go");
        assert_eq!(
            bound(&mut app, "field"),
            Some(handles[0]),
            "clearing one input must leave the others alone"
        );

        // The required one is refused by the same gate that refuses never
        // having bound it, and says the same thing.
        let mut refused = send(&app, |reply| SceneCommand::SetActor {
            id: actor,
            params: clear("field"),
            visible: None,
            subset: None,
            parents: None,
            reply,
        });
        app.update();
        let refusal = refused
            .try_recv()
            .expect("a reply")
            .expect_err("a required input cannot be cleared");
        assert!(
            matches!(refusal, SceneError::MissingInput { input, .. } if input == "field"),
            "expected a missing input, got {refusal:?}"
        );
        assert_eq!(
            bound(&mut app, "field"),
            Some(handles[0]),
            "a refused clear must leave the actor exactly as it was"
        );
    }

    /// Every kind the backend registered reaches a client, with the label and
    /// the parameters it declared.
    ///
    /// This is the only way a script learns what the running build can draw —
    /// there is no table on the client side to fall back on — so a kind that
    /// registers but does not list is invisible.
    #[test]
    fn listing_kinds_reports_every_registered_kind() {
        const PARAMS: &[registry::ParamSpec] = &[registry::ParamSpec {
            id: "size",
            label: "size",
            kind: registry::ParamKind::Float {
                default: 1.0,
                min: 0.0,
                max: 2.0,
                logarithmic: false,
            },
        }];

        let mut app = app();
        marker(&mut app);
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(registry::ActorKind {
                id: "sized",
                label: "Sized",
                params: PARAMS,
                apply: |_, _| {},
            });

        let mut listed = send(&app, |reply| SceneCommand::ListActorKinds { reply });
        app.update();
        let kinds = listed.try_recv().expect("a reply");

        let ids: Vec<&str> = kinds.iter().map(|kind| kind.id.as_str()).collect();
        assert_eq!(ids, vec!["marker", "sized"], "registration order");

        let sized = kinds.iter().find(|kind| kind.id == "sized").expect("listed");
        assert_eq!(sized.label, "Sized");
        assert_eq!(sized.params.len(), 1);
        assert_eq!(sized.params[0].id, "size");
    }

    /// One actor under two objects is drawn twice and stays one actor.
    ///
    /// The whole point of splitting an actor from its placements. Two actors
    /// binding the same arrays would look the same on screen and then have to
    /// be configured one at a time; this is one drawing, one mesh, and one
    /// thing to change.
    #[test]
    fn one_actor_is_drawn_under_every_object_it_names() {
        let mut app = app();
        marker(&mut app);

        let (first, second) = two_objects(&mut app);
        let actor = add(&mut app, vec![first, second]);

        let mut listed = send(&app, |reply| SceneCommand::ListActors {
            parent: None,
            reply,
        });
        app.update();
        let listing = listed.try_recv().expect("a reply").expect("a listing");
        assert_eq!(
            listing.iter().map(|a| a.id).collect::<Vec<_>>(),
            vec![actor],
            "two placements are still one actor"
        );
        assert_eq!(listing[0].parents, vec![first, second]);

        // And each object reports it, because each of them draws it.
        let mut objects = send(&app, |reply| SceneCommand::ListObjects { reply });
        app.update();
        assert_eq!(
            objects
                .try_recv()
                .expect("a reply")
                .iter()
                .map(|o| (o.id, o.actors.iter().map(|a| a.id).collect::<Vec<_>>()))
                .collect::<Vec<_>>(),
            vec![(first, vec![actor]), (second, vec![actor])]
        );
    }

    /// Deleting an object costs an actor that one appearance and nothing else.
    ///
    /// Actors used to be destroyed with the object they were under, which meant
    /// tidying up one of the three places a drawing appeared destroyed the
    /// drawing. The actor is not in the tree at all now — only its placements
    /// are — so a deletion reaches exactly one of them.
    #[test]
    fn deleting_one_object_leaves_the_actor_drawn_under_the_others() {
        let mut app = app();
        marker(&mut app);

        let (first, second) = two_objects(&mut app);
        let actor = add(&mut app, vec![first, second]);

        let mut deleted = send(&app, |reply| SceneCommand::DeleteObject {
            id: first,
            reply,
        });
        app.update();
        assert_eq!(
            deleted.try_recv().expect("a reply").objects,
            vec![first],
            "a deletion takes the object named and nothing else"
        );

        let mut listed = send(&app, |reply| SceneCommand::ListActors {
            parent: None,
            reply,
        });
        app.update();
        let listing = listed.try_recv().expect("a reply").expect("a listing");
        assert_eq!(
            listing
                .iter()
                .map(|a| (a.id, a.parents.clone()))
                .collect::<Vec<_>>(),
            vec![(actor, vec![second])],
            "the actor survives, drawn under what is left"
        );
        assert!(
            listing[0].visible,
            "and its own visibility setting is untouched"
        );
    }

    /// An actor under nothing draws nothing, and comes back when placed.
    ///
    /// No hidden node arranges this and nothing marks the actor: with the mesh
    /// on the placements, having none *is* being off screen.
    #[test]
    fn an_actor_with_no_parents_has_no_placements() {
        let mut app = app();
        marker(&mut app);

        let (first, second) = two_objects(&mut app);
        let actor = add(&mut app, vec![first]);

        let set = |app: &App, parents: Vec<u64>| {
            send(app, move |reply| SceneCommand::SetActor {
                id: actor,
                params: registry::ParamMap::default(),
                visible: None,
                subset: None,
                parents: Some(parents),
                reply,
            })
        };

        let mut cleared = set(&app, vec![]);
        app.update();
        assert!(
            cleared
                .try_recv()
                .expect("a reply")
                .expect("an actor")
                .parents
                .is_empty()
        );
        assert_eq!(placements(&mut app), 0, "nothing is drawn");

        let mut placed = set(&app, vec![second]);
        app.update();
        assert_eq!(
            placed
                .try_recv()
                .expect("a reply")
                .expect("an actor")
                .parents,
            vec![second],
            "and it draws again once it has somewhere to be"
        );
        assert_eq!(placements(&mut app), 1);
    }

    /// A filter kind that reads one array and writes two, so a test can tell
    /// "an output" from "the outputs" and check declaration order.
    fn passthrough(app: &mut App) {
        const PARAMS: &[registry::ParamSpec] = &[registry::ParamSpec {
            id: "values",
            label: "values",
            kind: registry::ParamKind::Array {
                dtypes: &[],
                shape: &[],
                required: true,
                structural: true,
            },
        }];
        const OUTPUTS: &[filter::OutputSpec] = &[
            filter::OutputSpec {
                id: "first",
                label: "first",
                kind: filter::OutputKind::Array {
                    dtype: Dtype::Uint8,
                    shape: &[0],
                },
            },
            filter::OutputSpec {
                id: "second",
                label: "second",
                kind: filter::OutputKind::Array {
                    dtype: Dtype::Uint8,
                    shape: &[0],
                },
            },
        ];

        app.world_mut()
            .resource_mut::<filter::FilterRegistry>()
            .register(filter::FilterKind {
                id: "passthrough",
                label: "passthrough",
                params: PARAMS,
                outputs: OUTPUTS,
                run: |_| filter::Products::new(),
            });
    }

    fn add_filter(
        app: &App,
        kind: &str,
        bound: u64,
    ) -> oneshot::Receiver<Result<FilterSummary, SceneError>> {
        let mut params = registry::ParamMap::new();
        params.insert("values".to_string(), registry::ParamValue::Data(bound));
        send(app, |reply| SceneCommand::AddFilter {
            kind: kind.into(),
            params,
            reply,
        })
    }

    /// One upload, and the handle it came back as.
    fn one_array(app: &mut App) -> u64 {
        let mut uploaded = send(app, |reply| SceneCommand::UploadData {
            arrays: vec![array("values", 4)],
            reply,
        });
        app.update();
        uploaded.try_recv().expect("a reply")[0].id
    }

    /// The reply carries usable handles, so the next call can bind one. Without
    /// this every caller would have to add a filter, wait, and ask again.
    #[test]
    fn adding_a_filter_allocates_a_handle_per_declared_output() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let names: Vec<&str> = summary
            .outputs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, vec!["first", "second"], "declaration order");

        let store = app.world().resource::<DataStore>();
        for (name, handle) in &summary.outputs {
            let held = store.array(*handle).expect("registered before the first run");
            assert_eq!(&held.meta.name, name);
        }
    }

    /// Naming a kind this build does not have is the caller's mistake, and the
    /// reply says how to find out what does exist.
    #[test]
    fn adding_an_unregistered_filter_kind_is_refused() {
        let mut app = app();
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "no-such-filter", values);
        app.update();
        assert!(matches!(
            added.try_recv().expect("a reply"),
            Err(SceneError::UnknownFilterKind { .. })
        ));
    }

    /// A filter that fed itself would never come to rest: each run rewrites an
    /// array that marks the next one stale, forever, with the app awake
    /// throughout. Refusing the binding is much cheaper than detecting the spin.
    #[test]
    fn a_filter_cannot_be_made_to_read_its_own_output() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");
        let (_, own_output) = summary.outputs[0].clone();

        let mut params = registry::ParamMap::new();
        params.insert("values".to_string(), registry::ParamValue::Data(own_output));
        let mut set = send(&app, |reply| SceneCommand::SetFilter {
            id: summary.id,
            params,
            reply,
        });
        app.update();

        assert!(matches!(
            set.try_recv().expect("a reply"),
            Err(SceneError::FilterCycle { .. })
        ));
    }

    /// The indirect case, which is the one a caller cannot see coming: two
    /// filters that each look reasonable on their own.
    #[test]
    fn a_cycle_through_another_filter_is_refused() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut first = add_filter(&app, "passthrough", values);
        app.update();
        let first = first.try_recv().expect("a reply").expect("added");

        // The second reads the first. Fine so far.
        let mut second = add_filter(&app, "passthrough", first.outputs[0].1);
        app.update();
        let second = second.try_recv().expect("a reply").expect("added");

        // Now point the first at the second, closing the loop.
        let mut params = registry::ParamMap::new();
        params.insert(
            "values".to_string(),
            registry::ParamValue::Data(second.outputs[0].1),
        );
        let mut set = send(&app, |reply| SceneCommand::SetFilter {
            id: first.id,
            params,
            reply,
        });
        app.update();

        assert!(matches!(
            set.try_recv().expect("a reply"),
            Err(SceneError::FilterCycle { .. })
        ));
    }

    /// Releasing an array a filter writes would leave it producing into nothing,
    /// which on screen is indistinguishable from a filter that has broken.
    #[test]
    fn a_filters_output_cannot_be_released_on_its_own() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");
        let generated = summary.outputs[0].1;

        let mut released = send(&app, |reply| SceneCommand::ReleaseData {
            // The upload alongside it, to show one refusal does not lose the
            // rest of the batch.
            ids: vec![generated, values],
            reply,
        });
        app.update();

        assert_eq!(released.try_recv().expect("a reply"), vec![values]);
        assert!(
            app.world().resource::<DataStore>().array(generated).is_some(),
            "still held, because the filter is still writing it"
        );
    }

    /// Removing the filter *is* how those handles go away.
    #[test]
    fn removing_a_filter_forgets_the_arrays_it_was_writing() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let mut removed = send(&app, |reply| SceneCommand::RemoveFilter {
            id: summary.id,
            reply,
        });
        app.update();
        assert!(removed.try_recv().expect("a reply"));

        let store = app.world().resource::<DataStore>();
        for (_, handle) in &summary.outputs {
            assert!(store.array(*handle).is_none(), "released with the filter");
        }
        assert!(store.array(values).is_some(), "the upload is untouched");
    }

    /// A listing is how a client rediscovers a scene it did not build, so it has
    /// to carry the output handles as well as the settings.
    #[test]
    fn listing_filters_reports_their_outputs() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let mut listed = send(&app, |reply| SceneCommand::ListFilters { reply });
        app.update();
        let filters = listed.try_recv().expect("a reply");

        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].id, summary.id);
        assert_eq!(filters[0].kind, "passthrough");
        assert_eq!(filters[0].outputs, summary.outputs);
    }
}
