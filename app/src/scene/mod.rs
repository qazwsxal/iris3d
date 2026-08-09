//! The scene: objects held in the Bevy world, and the command interface used
//! to get data into and out of them.
//!
//! Three layers, deliberately separated:
//!
//! - [`data`] — raw arrays as shared assets, plus what their bytes mean.
//! - [`dataset`] — what an object *is*: points, mesh, grid, molecule.
//! - [`actor`] — how it gets *drawn*, as separate entities so one
//!   dataset can be shown several ways at once.
//! - [`link`] — which object an actor draws, which is *not* the same
//!   question as where it sits.
//!
//! Objects form a tree. There is no separate group type: an object with no data
//! is a grouping node, and any object may parent any other, so a field can be
//! made to follow the mesh it belongs to.
//!
//! Nothing here knows about gRPC, and nothing here draws. A rendering backend
//! plugs in by consuming [`Actor`] and the dataset components — see
//! [`crate::draw`], which is one such backend and deliberately not the only
//! possible one.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};
use tokio::sync::oneshot;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::grpc::GrpcBridge;
use crate::redraw::KeepAwake;

pub mod actor;
pub mod data;
pub mod dataset;
pub mod ingest;
pub mod link;
pub mod registry;
pub mod subset;

// Only what other modules reach for. The rest stays available under its own
// module path — this is a binary crate, so unused re-exports are just noise.
pub use actor::ColorBy;
pub use data::{BufferMeta, DataArray, Dtype, NamedArray, NamedBuffer};
pub use dataset::{DatasetKind, MeshData, MoleculeData, PointCloud};
pub use link::{ActorOf, Actors};
pub use registry::{ActorKindId, ActorParams, ActorRegistry};
pub use subset::{Subset, SubsetEncoding};

/// Ceiling on how far the ancestor walk will climb before giving up. Guards
/// against a pre-existing malformed hierarchy sending validation into a loop.
const MAX_HIERARCHY_DEPTH: usize = 4096;

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<DataArray>().add_systems(
            Update,
            // Order matters throughout: the drain creates actors, the reaper
            // removes any a deletion orphaned before a backend can look at
            // them, and only then are style components derived from the
            // parameters the drain wrote.
            (
                apply_scene_commands,
                link::reap_orphaned_actors,
                registry::apply_actor_params,
            )
                .chain(),
        );
    }
}

/// An object in the scene. Paired with a [`UniqueID`] carrying its handle, at
/// most one dataset component, a [`Fields`](data::Fields) map, and child actor
/// entities.
#[derive(Component, Debug)]
pub struct SceneObject {
    pub name: String,
    /// Every array uploaded with the object, retained so the object can be
    /// described without dereferencing its contents. Empty for a grouping node.
    pub arrays: Vec<NamedArray>,
}

impl SceneObject {
    pub fn total_bytes(&self) -> u64 {
        self.arrays
            .iter()
            .filter_map(|array| array.meta.byte_length())
            .sum()
    }
}

/// A description of an object in the scene, without its contents.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub id: u64,
    pub name: String,
    pub kind: DatasetKind,
    pub buffers: Vec<BufferMeta>,
    pub total_bytes: u64,
    /// Everything currently drawing this object's data.
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
    /// The object whose data is drawn.
    pub source: u64,
    /// The object whose transform is inherited, usually the same as `source`.
    /// `None` when the actor has somehow been detached.
    pub parent: Option<u64>,
    pub params: registry::ParamMap,
    pub colour: ColorBy,
    pub visible: bool,
    /// How much of the source is drawn, or `None` for all of it. The selection
    /// values are not carried back — the caller sent them, and they can be
    /// large.
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
    /// Dataset kinds this can draw, by name.
    pub supports: Vec<String>,
    pub params: &'static [registry::ParamSpec],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    NoSuchObject(u64),
    NoSuchActor(u64),
    /// No backend registered a kind by that name, so nothing could draw it.
    UnknownKind(String),
    /// The kind exists but cannot draw this shape of data — ball-and-stick over
    /// a triangle mesh, say.
    KindNotSupported {
        kind: String,
        dataset: DatasetKind,
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
            SceneError::UnknownKind(kind) => write!(
                f,
                "no actor kind \"{kind}\" — ask ListActorKinds \
                 for the ones this build supports"
            ),
            SceneError::KindNotSupported { kind, dataset } => write!(
                f,
                "actor kind \"{kind}\" cannot draw {} data",
                dataset.as_str()
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
    InsertObject {
        name: String,
        buffers: Vec<NamedBuffer>,
        /// The client's declaration that the buffers are fields over a regular
        /// grid. `None` leaves the structure to be inferred from their names.
        grid: Option<dataset::GridData>,
        reply: oneshot::Sender<ObjectSummary>,
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
        /// When false, descendants are detached and become roots.
        recursive: bool,
        reply: oneshot::Sender<Deleted>,
    },

    /// Draws an object an additional way. Adds; never replaces.
    AddActor {
        /// The object whose data to draw.
        source: u64,
        /// `None` takes whatever the registry draws this dataset with.
        kind: Option<String>,
        /// The object whose transform to inherit. `None` means `source`.
        parent: Option<u64>,
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
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    RemoveActor {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    ListActors {
        /// Restrict to those drawing one object.
        source: Option<u64>,
        reply: oneshot::Sender<Result<Vec<ActorSummary>, SceneError>>,
    },
    ListActorKinds {
        reply: oneshot::Sender<Vec<KindSummary>>,
    },
}

/// What a deletion took with it.
#[derive(Debug, Default, Clone)]
pub struct Deleted {
    pub objects: Vec<u64>,
    /// Actors that were drawing the deleted objects, including any placed
    /// under an object that survives.
    pub actors: Vec<u64>,
}

/// Read-only view of the objects in the scene.
///
/// Yields [`Actors`] rather than `Children`: the two answer different
/// questions now, and every use here wants the actors drawing an object, not
/// the mix of actors and nested objects sitting under it. Walking the tree
/// needs a separate `Query<&Children>`.
type Objects<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static SceneObject,
        &'static DatasetKind,
        Option<&'static Actors>,
    ),
>;

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
        &'static ActorOf,
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
    &'a ActorOf,
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
    mut transforms: Query<&mut Transform>,
    objects: Objects,
    ids: Query<&UniqueID>,
    fields: Query<&data::Fields>,
    children: Query<&Children>,
    child_of: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    mut actors: ActorQuery,
    mut awake: ResMut<KeepAwake>,
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
            SceneCommand::InsertObject {
                name,
                buffers,
                grid,
                reply,
            } => {
                let ingested = ingest::ingest(buffers, grid, &mut arrays);
                let object = SceneObject {
                    name,
                    arrays: ingested.arrays,
                };
                let (id, summary) = spawn_object(
                    &mut commands,
                    &mut counter,
                    &registry,
                    &mut index,
                    object,
                    ingested.kind,
                    Some(ingested.fields),
                );
                match ingested.dataset {
                    ingest::Dataset::Points(points) => {
                        commands.entity(index[&id]).insert(points);
                    }
                    ingest::Dataset::Mesh(mesh) => {
                        commands.entity(index[&id]).insert(mesh);
                    }
                    ingest::Dataset::Grid(grid) => {
                        commands.entity(index[&id]).insert(grid);
                    }
                    ingest::Dataset::Molecule(molecule) => {
                        commands.entity(index[&id]).insert(molecule);
                    }
                    ingest::Dataset::Raw => {}
                }
                let _ = reply.send(summary);
            }

            SceneCommand::CreateObject { name, reply } => {
                let object = SceneObject {
                    name,
                    arrays: Vec::new(),
                };
                let (_, summary) = spawn_object(
                    &mut commands,
                    &mut counter,
                    &registry,
                    &mut index,
                    object,
                    DatasetKind::Empty,
                    None,
                );
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
                    .map(|(entity, id, object, kind, drawn_by)| {
                        let drawn = drawn_by
                            .into_iter()
                            .flat_map(|list| list.iter())
                            .filter_map(|entity| {
                                summarise_actor(actors.get(entity).ok()?, &ids, &arrays)
                            })
                            .collect();
                        let parent = effective_parent(entity, &pending_parent, &child_of)
                            .and_then(|p| ids.get(p).ok())
                            .map(|unique| unique.0);
                        summarise(id.0, object, *kind, drawn, parent)
                    })
                    .collect();
                listing.sort_by_key(|summary| summary.id);
                let _ = reply.send(listing);
            }

            SceneCommand::DeleteObject {
                id,
                recursive,
                reply,
            } => {
                let removed = delete_object(
                    &mut commands,
                    &mut index,
                    &mut drawn,
                    &objects,
                    &children,
                    &ids,
                    id,
                    recursive,
                );
                let _ = reply.send(removed);
            }

            SceneCommand::AddActor {
                source,
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
                    &objects,
                    &fields,
                    source,
                    kind,
                    parent,
                    params,
                    colour,
                    subset,
                    &mut arrays,
                );
                let _ = reply.send(result);
            }

            SceneCommand::SetActor {
                id,
                params,
                colour,
                visible,
                subset,
                reply,
            } => {
                let result = set_actor(
                    &registry,
                    &drawn,
                    &mut actors,
                    &ids,
                    &mut arrays,
                    id,
                    params,
                    colour,
                    visible,
                    subset,
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

            SceneCommand::ListActors { source, reply } => {
                let filter = match source {
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
                    .filter(|item| filter.is_none_or(|object| item.7.0 == object))
                    .filter_map(|item| summarise_actor(item, &ids, &arrays))
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
                        supports: DatasetKind::ALL
                            .iter()
                            .filter(|dataset| (kind.supports)(**dataset))
                            .map(|dataset| dataset.as_str().to_string())
                            .collect(),
                        params: kind.params,
                    })
                    .collect();
                let _ = reply.send(kinds);
            }
        }
    }
}

/// Adds a way of drawing an object.
#[allow(clippy::too_many_arguments)]
fn add_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &ActorRegistry,
    index: &HashMap<u64, Entity>,
    drawn: &mut HashMap<u64, Entity>,
    objects: &Objects,
    fields: &Query<&data::Fields>,
    source: u64,
    kind: Option<String>,
    parent: Option<u64>,
    params: registry::ParamMap,
    colour: Option<ColorBy>,
    subset: Option<subset::SubsetRequest>,
    arrays: &mut Assets<DataArray>,
) -> Result<ActorSummary, SceneError> {
    let source_entity = *index.get(&source).ok_or(SceneError::NoSuchObject(source))?;
    let parent_entity = match parent {
        Some(id) => *index.get(&id).ok_or(SceneError::NoSuchObject(id))?,
        // Drawn where the data already is, which is what almost every caller
        // wants and what an upload does.
        None => source_entity,
    };

    let dataset = objects
        .get(source_entity)
        .map(|(_, _, _, kind, _)| *kind)
        .map_err(|_| SceneError::NoSuchObject(source))?;

    let registered = match &kind {
        Some(name) => registry
            .get(name)
            .ok_or_else(|| SceneError::UnknownKind(name.clone()))?,
        None => registry
            .default_for(dataset)
            // Nothing registered can draw this shape of data, which is not the
            // caller naming something wrong.
            .ok_or_else(|| SceneError::KindNotSupported {
                kind: "default".into(),
                dataset,
            })?,
    };
    if !(registered.supports)(dataset) {
        return Err(SceneError::KindNotSupported {
            kind: registered.id.to_string(),
            dataset,
        });
    }

    // Unset parameters take the kind's default: this is a new actor, so there
    // is no previous value to preserve.
    let params = registered.normalise(&params);
    let colour = colour.unwrap_or_else(|| ColorBy {
        field: default_colour_field(fields.get(source_entity).ok()),
        ..default()
    });

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
        source_entity,
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
        "scene: object {source} also drawn as {} (actor {id}){}",
        registered.id,
        match parent {
            Some(parent) if parent != source => format!(", placed under object {parent}"),
            _ => String::new(),
        }
    );

    Ok(ActorSummary {
        id,
        kind: registered.id.to_string(),
        source,
        parent: Some(parent.unwrap_or(source)),
        params,
        colour,
        visible: true,
        subset: summarised_subset,
    })
}

/// Changes an existing actor, leaving anything unnamed alone.
#[allow(clippy::too_many_arguments)]
fn set_actor(
    registry: &ActorRegistry,
    drawn: &HashMap<u64, Entity>,
    actors: &mut ActorQuery,
    ids: &Query<&UniqueID>,
    arrays: &mut Assets<DataArray>,
    id: u64,
    params: registry::ParamMap,
    colour: Option<ColorBy>,
    visible: Option<bool>,
    subset: Option<Option<subset::SubsetRequest>>,
) -> Result<ActorSummary, SceneError> {
    let entity = *drawn.get(&id).ok_or(SceneError::NoSuchActor(id))?;
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
fn spawn_object(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &ActorRegistry,
    index: &mut HashMap<u64, Entity>,
    object: SceneObject,
    kind: DatasetKind,
    fields: Option<data::Fields>,
) -> (u64, ObjectSummary) {
    let id = counter.next();
    // Which kind this is depends entirely on what the backends registered, so
    // an upload of a dataset nothing can draw simply gets no actor rather than
    // one that silently does nothing.
    let default_kind = registry.default_for(kind);
    let colour = ColorBy {
        field: default_colour_field(fields.as_ref()),
        ..default()
    };
    // Taken before the object is moved into the world; the actor's handle is
    // only known after, so the summary is assembled at the end.
    let name = object.name.clone();
    let buffers: Vec<BufferMeta> = object
        .arrays
        .iter()
        .map(|array| array.meta.clone())
        .collect();
    let total_bytes = object.total_bytes();

    let spawned = commands
        .spawn((
            object,
            UniqueID(id),
            kind,
            fields.unwrap_or_default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    index.insert(id, spawned);

    // Give the object something to draw, so an upload is visible without a
    // follow-up call. Source and placement are the same object here; they only
    // differ once a client asks for an actor of one object under another.
    let drawn: Vec<ActorSummary> = default_kind
        .map(|default_kind| {
            let params = default_kind.defaults();
            let (actor, _) = link::spawn_actor(
                commands,
                counter,
                spawned,
                spawned,
                Subset::All,
                (
                    ActorKindId(default_kind.id),
                    ActorParams(params.clone()),
                    colour.clone(),
                ),
            );
            ActorSummary {
                id: actor,
                kind: default_kind.id.to_string(),
                source: id,
                parent: Some(id),
                params,
                colour,
                visible: true,
                // An upload draws all of what it uploaded.
                subset: None,
            }
        })
        .into_iter()
        .collect();

    let summary = ObjectSummary {
        id,
        name,
        kind,
        buffers,
        total_bytes,
        actors: drawn,
        parent: None,
    };

    info!(
        "scene: added {} object {} \"{}\" ({} arrays, {} bytes, {})",
        kind.as_str(),
        id,
        summary.name,
        summary.buffers.len(),
        summary.total_bytes,
        default_kind.map(|kind| kind.id).unwrap_or("no actor"),
    );

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

/// Removes an object, returning the handles actually removed.
///
/// Non-recursive is the default because `Children` is a `linked_spawn`
/// relationship: despawning an object would otherwise take every object
/// parented to it as well. Child *objects* are detached first; child *actors*
/// are left alone so they are despawned along with their object.
///
/// Actors *sourced* from a doomed object are taken down explicitly, because
/// they need not be parented to it. One parented elsewhere would survive the
/// despawn with its `Mesh3d` intact and go on drawing data that no longer
/// exists.
#[allow(clippy::too_many_arguments)]
fn delete_object(
    commands: &mut Commands,
    index: &mut HashMap<u64, Entity>,
    drawn: &mut HashMap<u64, Entity>,
    objects: &Objects,
    children: &Query<&Children>,
    ids: &Query<&UniqueID>,
    id: u64,
    recursive: bool,
) -> Deleted {
    let Some(entity) = index.get(&id).copied() else {
        return Deleted::default();
    };

    let mut doomed = vec![entity];

    if recursive {
        collect_descendants(entity, objects, children, &mut doomed);
    } else if let Ok(list) = children.get(entity) {
        for child in list.iter() {
            // Only detach objects. Actors belong to this object and should die
            // with it.
            if objects.contains(child) {
                commands.entity(child).remove::<ChildOf>();
            }
        }
    }

    let mut deleted = Deleted {
        objects: Vec::with_capacity(doomed.len()),
        actors: Vec::new(),
    };
    for object in doomed {
        if let Ok(unique) = ids.get(object) {
            deleted.objects.push(unique.0);
        }
        if let Ok((.., Some(drawn_by))) = objects.get(object) {
            for actor in drawn_by.iter() {
                if let Ok(unique) = ids.get(actor) {
                    deleted.actors.push(unique.0);
                }
                commands.entity(actor).despawn();
            }
        }
    }

    for handle in &deleted.objects {
        index.remove(handle);
    }
    for handle in &deleted.actors {
        drawn.remove(handle);
    }
    commands.entity(entity).despawn();

    info!(
        "scene: deleted object {id}{}{}",
        if recursive {
            format!(" and {} descendant(s)", deleted.objects.len() - 1)
        } else {
            String::new()
        },
        match deleted.actors.len() {
            0 => String::new(),
            n => format!(", and {n} actor(s) drawing them"),
        }
    );

    deleted
}

/// Appends every descendant *object* of `entity`, skipping actors, which are
/// children too but are not part of the object tree.
fn collect_descendants(
    entity: Entity,
    objects: &Objects,
    children: &Query<&Children>,
    out: &mut Vec<Entity>,
) {
    let Ok(list) = children.get(entity) else {
        return;
    };
    for child in list.iter() {
        if !objects.contains(child) {
            continue;
        }
        out.push(child);
        collect_descendants(child, objects, children, out);
    }
}

/// The field a new actor should colour by: the first scalar in name order, or
/// `None` when the object has no scalar to show.
///
/// Chosen once, here, so that whatever the UI displays is what is actually
/// drawn. Inferring it at draw time instead made "flat" in the UI mean
/// "coloured by something you cannot see the name of".
fn default_colour_field(fields: Option<&data::Fields>) -> Option<String> {
    let mut scalars: Vec<&String> = fields?
        .0
        .iter()
        .filter(|(_, field)| field.kind == data::FieldKind::Scalar)
        .map(|(name, _)| name)
        .collect();
    scalars.sort();
    scalars.first().map(|name| (*name).clone())
}

/// Describes one actor from its query item.
///
/// `None` when the source object has no handle, which should not happen and is
/// not worth inventing one for — such an actor is about to be reaped.
fn summarise_actor(
    (_, id, kind, params, colour, visibility, subset, source, parent): ActorItem<'_>,
    ids: &Query<&UniqueID>,
    arrays: &Assets<DataArray>,
) -> Option<ActorSummary> {
    Some(ActorSummary {
        id: id.0,
        kind: kind.0.to_string(),
        source: ids.get(source.0).ok()?.0,
        parent: parent
            .and_then(|link| ids.get(link.parent()).ok())
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
    kind: DatasetKind,
    actors: Vec<ActorSummary>,
    parent: Option<u64>,
) -> ObjectSummary {
    ObjectSummary {
        id,
        name: object.name.clone(),
        kind,
        buffers: object.arrays.iter().map(|a| a.meta.clone()).collect(),
        total_bytes: object.total_bytes(),
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
        app.init_resource::<GlobalIDCounter>();
        app.init_resource::<ActorRegistry>();
        app.init_resource::<KeepAwake>();
        app.init_resource::<GrpcBridge>();
        app.add_systems(Update, apply_scene_commands);
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
}
