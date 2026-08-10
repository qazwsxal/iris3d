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
//! is one such backend and deliberately not the only possible one.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};
use tokio::sync::oneshot;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::grpc::GrpcBridge;
use crate::redraw::KeepAwake;

pub mod actor;
pub mod data;
pub mod link;
pub mod registry;
pub mod subset;

// Only what other modules reach for. The rest stays available under its own
// module path — this is a binary crate, so unused re-exports are just noise.
pub use actor::ColorBy;
pub use data::{BufferMeta, DataArray, DataStore, Dtype, NamedBuffer};
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
            .add_systems(Startup, spawn_unplaced)
            .add_systems(
                Update,
                // Order matters: the drain creates actors, and only then are
                // style components and bindings derived from the parameters it
                // wrote.
                (apply_scene_commands, registry::apply_actor_params).chain(),
            );
    }
}

/// Where actors go when they have no object.
///
/// A hidden node, and the parent of every detached actor. Deleting an object
/// leaves its actors alive, and an actor with no object has nowhere to *be* —
/// its transform is an offset from whatever it is drawn under, not a place of
/// its own, so drawing one at that offset from the origin would put it
/// somewhere arbitrary. It must not draw at all until something adopts it.
///
/// A parent rather than writing `Visibility::Hidden` onto the actor itself.
/// That component is the client's own setting, reported back as
/// `ActorSummary::visible` and written by `SetActor`; overwriting it would both
/// destroy what the client asked for and let the next `SetActor { visible:
/// true }` put an unplaced actor back on screen. Inherited from here, the rule
/// cannot be broken by any command, and re-attaching restores exactly the
/// visibility the client last chose.
#[derive(Resource, Debug, Clone, Copy)]
pub struct Unplaced(pub Entity);

fn spawn_unplaced(mut commands: Commands) {
    let entity = commands
        .spawn((
            Name::new("unplaced actors"),
            Transform::default(),
            Visibility::Hidden,
        ))
        .id();
    commands.insert_resource(Unplaced(entity));
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

/// A held array, described without its contents.
#[derive(Debug, Clone)]
pub struct DataSummary {
    pub id: u64,
    pub meta: BufferMeta,
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
    /// The object it is drawn under, or `None` if it is detached — which is
    /// where deleting that object leaves it, drawing at its own transform
    /// until something attaches it again.
    pub parent: Option<u64>,
    pub params: registry::ParamMap,
    pub colour: ColorBy,
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
    /// No backend registered a kind by that name, so nothing could draw it.
    UnknownKind(String),
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
            SceneError::UnknownKind(kind) => write!(
                f,
                "no actor kind \"{kind}\" — ask ListActorKinds \
                 for the ones this build supports"
            ),
            SceneError::WouldCycle { object, parent } => write!(
                f,
                "object {object} cannot be parented to {parent}: {parent} is {object} \
                 or one of its descendants"
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
        /// The object to draw under, whose transform it inherits and whose
        /// lifetime it shares. *What* it draws is in `params`, as bindings.
        parent: u64,
        /// Which registered kind draws it. Named by the caller, always: the
        /// server has no opinion on how a dataset should look.
        kind: String,
        /// Partial. Anything unset takes the kind's **default**, there being no
        /// previous value.
        params: registry::ParamMap,
        /// `None` colours by the first scalar field, as an upload does.
        colour: Option<ColorBy>,
        /// `None` draws the whole dataset.
        subset: Option<subset::SubsetRequest>,
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    SetActor {
        id: u64,
        /// Partial. Anything unset keeps its **current** value — the opposite
        /// of [`AddActor`](Self::AddActor), because here there is one.
        params: registry::ParamMap,
        /// `None` leaves colouring alone; `Some` replaces it outright.
        colour: Option<ColorBy>,
        visible: Option<bool>,
        /// `None` leaves the selection alone; `Some(None)` clears it back to
        /// drawing everything. Absent and cleared have to be distinguishable,
        /// which is what the nesting buys.
        subset: Option<Option<subset::SubsetRequest>>,
        /// Move it under another object. `None` leaves it where it is —
        /// including detached, which is where a deleted parent leaves it.
        parent: Option<u64>,
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
/// An object's actors are its children that an [`ActorQuery`] matches — the
/// rest of its children are nested objects. One list, told apart by what the
/// entity carries, since an actor is a plain child now.
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
        &'static mut ColorBy,
        &'static mut Visibility,
        &'static mut Subset,
        Option<&'static ChildOf>,
    ),
    With<ActorKindId>,
>;

/// What [`ActorQuery`] yields when read rather than written.
type ActorItem<'a> = (
    Entity,
    &'a UniqueID,
    &'a ActorKindId,
    &'a ActorParams,
    &'a ColorBy,
    &'a Visibility,
    &'a Subset,
    Option<&'a ChildOf>,
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
    mut arrays: ResMut<Assets<DataArray>>,
    mut store: ResMut<DataStore>,
    mut transforms: Query<&mut Transform>,
    objects: Objects,
    ids: Query<&UniqueID>,
    children: Query<&Children>,
    child_of: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    mut actors: ActorQuery,
    mut awake: ResMut<KeepAwake>,
    unplaced: Res<Unplaced>,
) {
    let batch: Vec<SceneCommand> = std::iter::from_fn(|| bridge.try_recv().ok()).collect();
    if batch.is_empty() {
        return;
    }

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
                let summaries: Vec<DataSummary> = uploaded
                    .into_iter()
                    .map(|buffer| {
                        let id = counter.next();
                        let meta = buffer.meta;
                        let handle = arrays.add(DataArray {
                            dtype: meta.dtype,
                            shape: meta.shape.clone(),
                            data: buffer.data,
                        });
                        store.insert(id, meta.clone(), handle);
                        DataSummary { id, meta }
                    })
                    .collect();
                info!(
                    "scene: took in {} array(s): {}",
                    summaries.len(),
                    summaries
                        .iter()
                        .map(|array| format!("{}={}", array.id, array.meta.name))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let _ = reply.send(summaries);
            }

            SceneCommand::ListData { reply } => {
                let listing = store
                    .iter()
                    .map(|(id, array)| DataSummary {
                        id,
                        meta: array.meta.clone(),
                    })
                    .collect();
                let _ = reply.send(listing);
            }

            SceneCommand::ReleaseData { ids, reply } => {
                let released: Vec<u64> = ids.into_iter().filter(|id| store.remove(*id)).collect();
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
                        // An object's actors are the children the actor query
                        // matches; the rest of its children are nested objects.
                        let drawn = children
                            .get(entity)
                            .into_iter()
                            .flat_map(|list| list.iter())
                            .filter_map(|child| {
                                summarise_actor(
                                    actors.get(child).ok()?,
                                    &pending_parent,
                                    &ids,
                                    &arrays,
                                )
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
                    unplaced.0,
                    &objects,
                    &children,
                    id,
                );
                let _ = reply.send(removed);
            }

            SceneCommand::AddActor {
                kind,
                parent,
                params,
                colour,
                subset,
                reply,
            } => {
                let result = add_actor(
                    &mut commands,
                    &mut counter,
                    &registry,
                    &index,
                    &mut drawn,
                    parent,
                    kind,
                    params,
                    colour,
                    subset,
                    &mut arrays,
                    &store,
                );
                let _ = reply.send(result);
            }

            SceneCommand::SetActor {
                id,
                params,
                colour,
                visible,
                subset,
                parent,
                reply,
            } => {
                let result = set_actor(
                    &mut commands,
                    &registry,
                    &drawn,
                    &index,
                    &mut pending_parent,
                    &mut actors,
                    &ids,
                    &mut arrays,
                    id,
                    params,
                    colour,
                    visible,
                    subset,
                    parent,
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
                    .filter(|item| {
                        filter
                            .is_none_or(|object| item.7.is_some_and(|link| link.parent() == object))
                    })
                    .filter_map(|item| summarise_actor(item, &pending_parent, &ids, &arrays))
                    .collect();
                listing.sort_by_key(|summary| summary.id);
                let _ = reply.send(Ok(listing));
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

/// Checks that every array an actor kind reads is bound, and bound to something
/// it can actually read.
///
/// Separate from [`registry::ParamKind::sanitise`] on purpose. Sanitising judges
/// a value on its own and runs wherever a parameter is written; this needs the
/// [`DataStore`] to see what a handle points at, and the store is not reachable
/// from all of those places. So one answers "is this the right sort of value"
/// and the other "is that particular array the right shape".
fn check_bindings(
    kind: &registry::ActorKind,
    params: &registry::ParamMap,
    store: &DataStore,
) -> Result<(), SceneError> {
    for spec in kind.inputs() {
        let registry::ParamKind::Array { required, .. } = spec.kind else {
            continue;
        };
        match registry::data(params, spec.id) {
            Some(id) => {
                let array = store.get(id).ok_or(SceneError::NoSuchData(id))?;
                spec.kind
                    .accepts(&array.meta)
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

/// Adds a way of drawing, at a place in the tree.
#[allow(clippy::too_many_arguments)]
fn add_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &ActorRegistry,
    index: &HashMap<u64, Entity>,
    drawn: &mut HashMap<u64, Entity>,
    parent: u64,
    kind: String,
    params: registry::ParamMap,
    colour: Option<ColorBy>,
    subset: Option<subset::SubsetRequest>,
    arrays: &mut Assets<DataArray>,
    store: &DataStore,
) -> Result<ActorSummary, SceneError> {
    let parent_entity = *index.get(&parent).ok_or(SceneError::NoSuchObject(parent))?;

    // The caller names the kind. There is no default to fall back on: which
    // representation suits some data is a judgement, and the server has no
    // basis for it beyond the order its backends registered in.
    //
    // Nor is there a check that the kind suits the object any more. It used to
    // ask `supports(DatasetKind)`, which only meant anything while an actor took
    // its data from the object it hung under. An actor binds its own arrays now,
    // so what the object holds — usually nothing — says nothing about what can
    // draw there. `check_bindings` is the real gate, and it asks about the data
    // rather than about the node.
    let registered = registry
        .get(&kind)
        .ok_or_else(|| SceneError::UnknownKind(kind.clone()))?;

    // Unset parameters take the kind's default: this is a new actor, so there
    // is no previous value to preserve.
    let params = registered.normalise(&params);
    check_bindings(registered, &params, store)?;
    let colour = colour.unwrap_or_default();

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

    let (id, entity) = link::spawn_actor(
        commands,
        counter,
        parent_entity,
        subset,
        (
            ActorKindId(registered.id),
            ActorParams(params.clone()),
            colour.clone(),
        ),
    );
    drawn.insert(id, entity);

    info!(
        "scene: actor {id} draws {} under object {parent}",
        registered.id
    );

    Ok(ActorSummary {
        id,
        kind: registered.id.to_string(),
        parent: Some(parent),
        params,
        colour,
        visible: true,
        subset: summarised_subset,
    })
}

/// Changes an existing actor, leaving anything unnamed alone.
#[allow(clippy::too_many_arguments)]
fn set_actor(
    commands: &mut Commands,
    registry: &ActorRegistry,
    drawn: &HashMap<u64, Entity>,
    index: &HashMap<u64, Entity>,
    pending_parent: &mut HashMap<Entity, Option<Entity>>,
    actors: &mut ActorQuery,
    ids: &Query<&UniqueID>,
    arrays: &mut Assets<DataArray>,
    id: u64,
    params: registry::ParamMap,
    colour: Option<ColorBy>,
    visible: Option<bool>,
    subset: Option<Option<subset::SubsetRequest>>,
    parent: Option<u64>,
) -> Result<ActorSummary, SceneError> {
    let entity = *drawn.get(&id).ok_or(SceneError::NoSuchActor(id))?;

    // Resolved before anything is written, so a bad handle changes nothing.
    let moving_to = parent
        .map(|id| index.get(&id).copied().ok_or(SceneError::NoSuchObject(id)))
        .transpose()?;

    let Ok(mut item) = actors.get_mut(entity) else {
        // In `drawn` but not yet in the world: added earlier in this same
        // drain, so the query has not seen it. Nothing is lost by asking again
        // next tick, and guessing would mean writing to an entity blind.
        return Err(SceneError::NoSuchActor(id));
    };

    let registered = registry
        .get(item.2.0)
        .ok_or_else(|| SceneError::UnknownKind(item.2.0.to_string()))?;

    // Merge rather than replace: a client changing one setting should not have
    // to restate the others, and omitting them must not silently reset them.
    for (key, value) in params {
        let Some(value) = registered
            .spec(&key)
            .and_then(|spec| spec.kind.sanitise(value))
        else {
            warn!("scene: actor {id} has no parameter \"{key}\" of that type");
            continue;
        };
        item.3.0.insert(key, value);
    }

    if let Some(colour) = colour {
        *item.4 = colour;
    }
    if let Some(visible) = visible {
        *item.5 = if visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    if let Some(subset) = subset {
        *item.6 = subset.map_or(Subset::All, |request| request.into_subset(arrays));
    }

    if let Some(parent) = moving_to {
        // No cycle check, unlike `set_parent`. An actor is a leaf — nothing is
        // ever parented under one — so no placement of it can close a loop.
        //
        // Its local transform is kept as it stands, which means it moves on
        // screen if the new parent sits somewhere else. That is what an
        // actor's transform *is* — an offset from whatever it is drawn under,
        // not a place of its own — so there is nothing to preserve here, and
        // no equivalent of `set_parent`'s world-transform arithmetic.
        //
        // Adopting one out of `Unplaced` is the same write, and is how a
        // detached actor starts drawing again.
        commands.entity(entity).insert(ChildOf(parent));
        pending_parent.insert(entity, Some(parent));
    }

    summarise_actor(
        actors.get(entity).expect("the entity was just written"),
        pending_parent,
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
/// out of whatever order the backends happened to register in. A client that
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
/// Deletes exactly what was named, and nothing else. Every child survives it —
/// but they do not all survive it the same way, because an object and an actor
/// are not the same kind of thing.
///
/// A child *object* becomes a root. Its transform is a place, so it still has
/// one with nothing above it.
///
/// A child *actor* moves to [`Unplaced`] and stops drawing. Its transform is an
/// offset from whatever it is drawn under, so with nothing above it there is no
/// answer to where it belongs, and drawing it at that offset from the origin
/// would put it somewhere no one chose. It keeps its arrays, its parameters and
/// its own visibility setting, and waits for `SetActor`'s `parent`.
///
/// Actors used to die here instead. That made sense while an actor drew a
/// source object's data, because losing the object left it drawing nothing;
/// the `reap_orphaned_actors` sweep existed for exactly that. An actor binds
/// arrays that outlive every node now, so one whose parent is deleted is still
/// completely defined and worth keeping. `RemoveActor` is what destroys one.
fn delete_object(
    commands: &mut Commands,
    index: &mut HashMap<u64, Entity>,
    pending_parent: &mut HashMap<Entity, Option<Entity>>,
    unplaced: Entity,
    objects: &Objects,
    children: &Query<&Children>,
    id: u64,
) -> Deleted {
    let Some(entity) = index.get(&id).copied() else {
        return Deleted::default();
    };

    if let Ok(list) = children.get(entity) {
        for child in list.iter() {
            // Anything here that is not an object is an actor drawn under it.
            let moved_to = if objects.contains(child) {
                commands.entity(child).remove::<ChildOf>();
                None
            } else {
                commands.entity(child).insert(ChildOf(unplaced));
                Some(unplaced)
            };
            // Both are queued, so anything listing the scene later in this same
            // drain would otherwise report a parent about to be despawned.
            pending_parent.insert(child, moved_to);
        }
    }

    // The handle is what the lookup above matched on, so there is nothing to
    // read back off the entity.
    index.remove(&id);
    commands.entity(entity).despawn();
    let deleted = Deleted { objects: vec![id] };

    info!("scene: deleted object {id}; its children outlived it");

    deleted
}

/// Describes one actor from its query item.
///
/// `Option` only so callers can filter with `?`; an actor always describes.
fn summarise_actor(
    (entity, id, kind, params, colour, visibility, subset, parent): ActorItem<'_>,
    pending_parent: &HashMap<Entity, Option<Entity>>,
    ids: &Query<&UniqueID>,
    arrays: &Assets<DataArray>,
) -> Option<ActorSummary> {
    // Detachments and moves made earlier this drain are still queued, so the
    // link on the entity is the one from before the batch. Reporting that would
    // have `SetActor` answer with the parent it was just asked to replace.
    let parent = match pending_parent.get(&entity) {
        Some(pending) => *pending,
        None => parent.map(|link| link.parent()),
    };

    Some(ActorSummary {
        id: id.0,
        kind: kind.0.to_string(),
        parent: parent
            .and_then(|parent| ids.get(parent).ok())
            .map(|unique| unique.0),
        params: params.0.clone(),
        colour: colour.clone(),
        // What this actor was told; an object hidden above it still hides it
        // on screen, which is `InheritedVisibility`'s business.
        visible: *visibility != Visibility::Hidden,
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
        app.init_resource::<DataStore>();
        app.init_resource::<GlobalIDCounter>();
        app.init_resource::<ActorRegistry>();
        app.init_resource::<KeepAwake>();
        app.init_resource::<GrpcBridge>();
        // The parent detached actors are parked under. A Startup system in the
        // real plugin; run here directly, since these tests never run Startup.
        app.add_systems(Startup, spawn_unplaced);
        app.add_systems(Update, apply_scene_commands);
        app.update();
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

    fn array(name: &str, bytes: usize) -> NamedBuffer {
        NamedBuffer {
            meta: BufferMeta {
                name: name.into(),
                dtype: Dtype::Uint8,
                shape: vec![bytes as u64],
            },
            data: vec![0; bytes],
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
                .map(|a| a.meta.name.as_str())
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

    /// Deleting an object detaches the actors drawn under it, parks them where
    /// they cannot draw, and lets one be drawn under some other object.
    ///
    /// They used to be destroyed, which made deletion inconsistent with itself:
    /// a child object survived and a child actor did not, so deleting a
    /// grouping node quietly took work with it. An actor binds arrays that
    /// outlive every node, so one whose parent is gone still knows everything
    /// it needs to draw and is only missing somewhere to be — and until it has
    /// somewhere, it must not be drawn anywhere.
    #[test]
    fn an_actor_outlives_the_object_it_was_drawn_under() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(registry::ActorKind {
                id: "marker",
                label: "Marker",
                params: &[],
                apply: |_, _| {},
            });

        let mut first = create(&app, "first");
        let mut second = create(&app, "second");
        app.update();
        let first = first.try_recv().expect("a reply").id;
        let second = second.try_recv().expect("a reply").id;

        let mut added = send(&app, |reply| SceneCommand::AddActor {
            parent: first,
            kind: "marker".into(),
            params: registry::ParamMap::default(),
            colour: None,
            subset: None,
            reply,
        });
        app.update();
        let actor = added.try_recv().expect("a reply").expect("an actor").id;

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
            listing.iter().map(|a| (a.id, a.parent)).collect::<Vec<_>>(),
            vec![(actor, None)],
            "the actor has to survive its object, detached"
        );
        assert!(
            listing[0].visible,
            "detaching must not touch the client's own visibility setting"
        );

        // But parked where nothing draws it. It has no object, so its transform
        // is an offset from nothing and there is no honest place to put it.
        let entity = *app.world().resource::<Unplaced>();
        let parked = app
            .world()
            .iter_entities()
            .find(|e| e.get::<UniqueID>().is_some_and(|id| id.0 == actor))
            .expect("the actor entity");
        assert_eq!(
            parked.get::<ChildOf>().map(|link| link.parent()),
            Some(entity.0),
            "a detached actor belongs to the unplaced node"
        );
        assert_eq!(
            app.world().get::<Visibility>(entity.0),
            Some(&Visibility::Hidden),
            "which is hidden, so nothing under it is drawn"
        );

        // And it goes back into the scene without being rebuilt.
        let mut moved = send(&app, |reply| SceneCommand::SetActor {
            id: actor,
            params: registry::ParamMap::default(),
            colour: None,
            visible: None,
            subset: None,
            parent: Some(second),
            reply,
        });
        app.update();
        assert_eq!(
            moved.try_recv().expect("a reply").expect("an actor").parent,
            Some(second),
            "a detached actor has to be attachable to another object"
        );
    }
}
