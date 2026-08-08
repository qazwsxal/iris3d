//! The scene: objects held in the Bevy world, and the command interface used
//! to get data into and out of them.
//!
//! Three layers, deliberately separated:
//!
//! - [`data`] — raw arrays as shared assets, plus what their bytes mean.
//! - [`dataset`] — what an object *is*: points, mesh, grid, molecule.
//! - [`representation`] — how it gets *drawn*, as child entities so one dataset
//!   can be shown several ways at once.
//!
//! Objects form a tree. There is no separate group type: an object with no data
//! is a grouping node, and any object may parent any other, so a field can be
//! made to follow the mesh it belongs to.
//!
//! Nothing here knows about gRPC, and nothing here draws. A rendering backend
//! plugs in by consuming [`Representation`] and the dataset components; no
//! backend is chosen yet.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};
use tokio::sync::oneshot;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::grpc::GrpcBridge;

pub mod data;
pub mod dataset;
pub mod ingest;
pub mod representation;

// Only what other modules reach for. The rest stays available under its own
// module path — this is a binary crate, so unused re-exports are just noise.
pub use data::{BufferMeta, DataArray, Dtype, NamedArray, NamedBuffer};
pub use dataset::{DatasetKind, MeshData, MoleculeData, PointCloud};
pub use representation::{ColorBy, Representation};

/// Ceiling on how far the ancestor walk will climb before giving up. Guards
/// against a pre-existing malformed hierarchy sending validation into a loop.
const MAX_HIERARCHY_DEPTH: usize = 4096;

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<DataArray>()
            .add_systems(Update, apply_scene_commands);
    }
}

/// An object in the scene. Paired with a [`UniqueID`] carrying its handle, at
/// most one dataset component, a [`Fields`](data::Fields) map, and child
/// representation entities.
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
    /// Names of the representations currently drawing this object.
    pub representations: Vec<String>,
    /// Parent in the scene tree, `None` for a root.
    pub parent: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    NoSuchObject(u64),
    /// The requested parent is the object itself or one of its descendants.
    ///
    /// Rejecting this is not optional: Bevy's transform propagation *panics* on
    /// a hierarchy cycle, so allowing one would let a client crash the
    /// application with two calls.
    WouldCycle { object: u64, parent: u64 },
}

impl Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::NoSuchObject(id) => write!(f, "no object with handle {id}"),
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
        reply: oneshot::Sender<Vec<u64>>,
    },
}

/// Read-only view of the objects in the scene.
type Objects<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static SceneObject,
        &'static DatasetKind,
        Option<&'static Children>,
    ),
>;

/// Drains commands submitted from outside the ECS and applies them to the
/// world. Replies are best-effort: a caller that has hung up is not an error.
///
/// Public so rendering backends can order themselves after it and pick up new
/// representations on the frame they appear.
///
/// Structural changes go through Bevy's deferred `Commands`, so the queries here
/// still show the pre-tick hierarchy. Parent changes made earlier in this same
/// drain are therefore tracked in `pending_parent` and consulted during cycle
/// validation — without that, two `SetParent` commands arriving in one tick
/// could each look safe in isolation and together form a cycle.
pub fn apply_scene_commands(
    mut commands: Commands,
    bridge: Res<GrpcBridge>,
    mut counter: ResMut<GlobalIDCounter>,
    mut arrays: ResMut<Assets<DataArray>>,
    mut transforms: Query<&mut Transform>,
    objects: Objects,
    ids: Query<&UniqueID>,
    child_of: Query<&ChildOf>,
    globals: Query<&GlobalTransform>,
    representations: Query<&Representation>,
) {
    let batch: Vec<SceneCommand> = std::iter::from_fn(|| bridge.try_recv().ok()).collect();
    if batch.is_empty() {
        return;
    }

    let mut index: HashMap<u64, Entity> = objects.iter().map(|(e, id, ..)| (id.0, e)).collect();
    let mut pending_parent: HashMap<Entity, Option<Entity>> = HashMap::new();

    for command in batch {
        match command {
            SceneCommand::InsertObject {
                name,
                buffers,
                reply,
            } => {
                let ingested = ingest::ingest(buffers, &mut arrays);
                let object = SceneObject {
                    name,
                    arrays: ingested.arrays,
                };
                let (id, summary) = spawn_object(
                    &mut commands,
                    &mut counter,
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
                    .map(|(entity, id, object, kind, children)| {
                        let drawn = children
                            .into_iter()
                            .flatten()
                            .filter_map(|child| representations.get(*child).ok())
                            .map(|representation| representation.as_str().to_string())
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
                    &objects,
                    &ids,
                    id,
                    recursive,
                );
                let _ = reply.send(removed);
            }
        }
    }
}

/// Spawns an object entity, its default representation, and registers its
/// handle. Returns the handle and a summary of the new object.
fn spawn_object(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    index: &mut HashMap<u64, Entity>,
    object: SceneObject,
    kind: DatasetKind,
    fields: Option<data::Fields>,
) -> (u64, ObjectSummary) {
    let id = counter.next();
    let default_representation = Representation::default_for(kind);
    let colour_field = default_colour_field(fields.as_ref());
    let summary = summarise(
        id,
        &object,
        kind,
        default_representation
            .iter()
            .map(|r| r.as_str().to_string())
            .collect(),
        None,
    );

    let mut entity = commands.spawn((
        object,
        UniqueID(id),
        kind,
        fields.unwrap_or_default(),
        Transform::default(),
        Visibility::default(),
    ));

    // Give the object something to draw, so an upload is visible without a
    // follow-up call.
    if let Some(representation) = default_representation.clone() {
        entity.with_children(|object| {
            // Transform and Visibility make the child a full participant in the
            // hierarchy, so it inherits the object's placement and visibility.
            // Bevy's `Mesh3d` requires `Transform` but *not* `Visibility`, so a
            // backend that only adds `Mesh3d` here would end up with an entity
            // the visibility systems never collect.
            object.spawn((
                representation,
                ColorBy {
                    field: colour_field,
                    ..default()
                },
                Transform::default(),
                Visibility::default(),
            ));
        });
    }

    let spawned = entity.id();
    index.insert(id, spawned);

    info!(
        "scene: added {} object {} \"{}\" ({} arrays, {} bytes, {})",
        kind.as_str(),
        id,
        summary.name,
        summary.buffers.len(),
        summary.total_bytes,
        default_representation
            .as_ref()
            .map(|r| r.as_str())
            .unwrap_or("no representation"),
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
        // GlobalTransform is whatever the last propagation produced, so an
        // object created earlier in this same tick still reads as identity.
        let world = globals.get(entity).copied().unwrap_or_default();
        let parent_world = parent_entity
            .and_then(|p| globals.get(p).ok().copied())
            .unwrap_or_default();
        if let Ok(mut local) = transforms.get_mut(entity) {
            *local = world.reparented_to(&parent_world);
        }
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
/// parented to it as well. Child *objects* are detached first; child
/// *representations* are left alone so they are despawned along with their
/// object.
fn delete_object(
    commands: &mut Commands,
    index: &mut HashMap<u64, Entity>,
    objects: &Objects,
    ids: &Query<&UniqueID>,
    id: u64,
    recursive: bool,
) -> Vec<u64> {
    let Some(entity) = index.get(&id).copied() else {
        return Vec::new();
    };

    let mut removed = vec![id];

    if recursive {
        collect_descendants(entity, objects, ids, &mut removed);
    } else if let Ok((_, _, _, _, Some(children))) = objects.get(entity) {
        for child in children.iter() {
            // Only detach objects. Representations belong to this object and
            // should die with it.
            if objects.contains(child) {
                commands.entity(child).remove::<ChildOf>();
            }
        }
    }

    for handle in &removed {
        index.remove(handle);
    }
    commands.entity(entity).despawn();

    info!(
        "scene: deleted object {id}{}",
        if recursive {
            format!(" and {} descendant(s)", removed.len() - 1)
        } else {
            String::new()
        }
    );

    removed
}

fn collect_descendants(entity: Entity, objects: &Objects, ids: &Query<&UniqueID>, out: &mut Vec<u64>) {
    let Ok((_, _, _, _, Some(children))) = objects.get(entity) else {
        return;
    };
    for child in children.iter() {
        if !objects.contains(child) {
            continue;
        }
        if let Ok(unique) = ids.get(child) {
            out.push(unique.0);
        }
        collect_descendants(child, objects, ids, out);
    }
}

/// The field a new representation should colour by: the first scalar in name
/// order, or `None` when the object has no scalar to show.
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

fn summarise(
    id: u64,
    object: &SceneObject,
    kind: DatasetKind,
    representations: Vec<String>,
    parent: Option<u64>,
) -> ObjectSummary {
    ObjectSummary {
        id,
        name: object.name.clone(),
        kind,
        buffers: object.arrays.iter().map(|a| a.meta.clone()).collect(),
        total_bytes: object.total_bytes(),
        representations,
        parent,
    }
}
