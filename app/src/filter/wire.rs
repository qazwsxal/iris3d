//! Creating, configuring and removing filters from a client.
//!
//! The actor half of this lives in [`scene`](crate::scene) beside the commands
//! themselves. This half is here because what a filter *is* belongs with the
//! filters, and `scene` only routes: it drains one queue and hands each command
//! to whoever understands it.
//!
//! Everything mirrors the actor path deliberately. `AddFilter` takes a partial
//! map and fills the rest from the kind's defaults; `SetFilter` takes a partial
//! map and merges it over what is already there. A client that has learned one
//! has learned the other.
//!
//! # Output handles are minted here
//!
//! Adding a filter allocates one handle per declared output from the same
//! sequence everything else uses, and puts an **empty** array in
//! [`DataStore`] under each. So the reply already carries handles a caller can
//! bind, and it need not wait for a first run to find out what they are.

use bevy::ecs::system::SystemParam;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::PrimitiveTopology;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::scene::data::{BufferMeta, GeometryMeta};
use crate::scene::registry::{ParamMap, ParamSpec};
use crate::scene::{DataArray, DataStore, Dtype, SceneError};

use super::{
    FilterKindId, FilterParams, FilterProblem, FilterRegistry, Generation, OutputKind, OutputSpec,
    Outputs,
};

/// The mesh a geometry output starts as: **one degenerate triangle**, which
/// draws nothing.
///
/// Positions and indices are present rather than left off, so the layout a
/// consumer's pipeline specialises over is the one it will keep once the filter
/// has run. A mesh with no attributes at all specialises to nothing, and the
/// error names a missing `Vertex_Position` — a confusing way to say "it has not
/// run yet".
///
/// Three coincident vertices rather than none, which is not a nicety.
/// `MeshAllocator` sub-allocates every mesh into a shared slab, and a
/// **zero-length** buffer gets a key it never allocates space for — so the first
/// write into it logs `Use-after-free: attempted to copy element data for an
/// unallocated key`, twice, once for the vertices and once for the indices.
/// Rendering survives it, which is what makes it worth pinning down here: the
/// only symptom was two red lines during startup. A triangle of zero area is a
/// real allocation and rasterises nothing.
fn empty_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; 3]);
    mesh.insert_indices(bevy::mesh::Indices::U32(vec![0, 1, 2]));
    mesh
}

/// Mutable view of the filter entities, for the same reason
/// [`ActorQuery`](crate::scene) is one query: a read-only query over the same
/// components as a `&mut` one is a conflict Bevy rejects at schedule init.
pub type FilterQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static FilterKindId,
        &'static mut FilterParams,
        &'static Outputs,
        Option<&'static FilterProblem>,
    ),
    With<FilterKindId>,
>;

/// The filter half of the world, as one system parameter.
///
/// Bundled rather than passed as two, because the scene's command drain was
/// already at Bevy's ceiling of sixteen system parameters and adding a registry
/// and a query outright pushed it over. Grouping them is also the honest
/// description: these two are always wanted together, by the one system that
/// routes filter commands.
#[derive(SystemParam)]
pub struct Filters<'w, 's> {
    pub registry: Res<'w, FilterRegistry>,
    pub entities: FilterQuery<'w, 's>,
}

/// One filter, described for a client.
#[derive(Debug, Clone)]
pub struct FilterSummary {
    pub id: u64,
    /// Registered kind id — see [`FilterRegistry`].
    pub kind: String,
    pub params: ParamMap,
    /// Where each declared output can be found, in declaration order so a
    /// listing is stable between calls. **This is what a caller binds.**
    pub outputs: Vec<(String, u64)>,
    /// Why the last run did not do what was asked.
    ///
    /// `None` means the last run was fine — or, for a filter that has just been
    /// created, that it has not run yet. The two are deliberately not
    /// distinguished: a filter with no complaint and no output is a filter whose
    /// answer has not arrived, and a client waiting on one polls the data rather
    /// than this.
    pub problem: Option<String>,
}

/// A way of deriving data, described for a client.
#[derive(Debug, Clone)]
pub struct FilterKindSummary {
    pub id: String,
    pub label: String,
    pub params: &'static [ParamSpec],
    pub outputs: &'static [OutputSpec],
}

/// Everything the drain needs to know about which filter reads what.
///
/// Built once per drain and kept up to date as commands are applied, so two
/// commands arriving in one tick see each other. Without that, adding a filter
/// and then pointing another at it would look like two unrelated calls and the
/// cycle check would miss the pair.
pub struct Graph {
    /// Data handle to the filter that writes it.
    producer: HashMap<u64, u64>,
    /// Filter to the data handles it reads.
    reads: HashMap<u64, Vec<u64>>,
}

impl Graph {
    pub fn build(registry: &FilterRegistry, filters: &FilterQuery) -> Self {
        let mut graph = Self {
            producer: HashMap::new(),
            reads: HashMap::new(),
        };
        for (_, id, kind, params, outputs, _) in filters.iter() {
            for handle in outputs.0.values() {
                graph.producer.insert(*handle, id.0);
            }
            graph.record_reads(registry, id.0, kind.0, &params.0);
        }
        graph
    }

    fn record_reads(
        &mut self,
        registry: &FilterRegistry,
        id: u64,
        kind: &str,
        params: &ParamMap,
    ) {
        let Some(registered) = registry.get(kind) else {
            return;
        };
        let reads = registered
            .inputs()
            .filter_map(|spec| crate::scene::registry::data(params, spec.id))
            .collect();
        self.reads.insert(id, reads);
    }

    /// Whether `filter` reading `handle` would close a loop.
    ///
    /// Walks *upstream* from the handle: whoever writes it, whatever they read,
    /// and so on. Reaching `filter` means its own output is somewhere in its own
    /// inputs.
    ///
    /// A cycle is not a slow render, it is a filter graph that can never come to
    /// rest: each run rewrites an array that marks the next one stale, forever,
    /// with the app awake the whole time. Cheaper to refuse the binding than to
    /// detect the spin afterwards.
    fn would_cycle(&self, filter: u64, handle: u64) -> bool {
        let mut seen: Vec<u64> = Vec::new();
        let mut queue = vec![handle];
        while let Some(handle) = queue.pop() {
            let Some(upstream) = self.producer.get(&handle).copied() else {
                // An uploaded array. Nothing produced it, so the walk ends.
                continue;
            };
            if upstream == filter {
                return true;
            }
            if seen.contains(&upstream) {
                continue;
            }
            seen.push(upstream);
            if let Some(reads) = self.reads.get(&upstream) {
                queue.extend(reads.iter().copied());
            }
        }
        false
    }

    /// The filter that writes this handle, if any. Used to refuse forgetting an
    /// array something is still generating.
    pub fn producer_of(&self, handle: u64) -> Option<u64> {
        self.producer.get(&handle).copied()
    }
}

/// Adds a filter, allocating a handle for each of its declared outputs.
///
/// Cannot create a cycle, whatever it binds: its outputs are minted in this
/// call, so nothing existing can be reading them yet.
#[allow(clippy::too_many_arguments)]
pub fn add(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &FilterRegistry,
    graph: &mut Graph,
    arrays: &mut Assets<DataArray>,
    meshes: &mut Assets<Mesh>,
    store: &mut DataStore,
    kind: String,
    params: ParamMap,
) -> Result<FilterSummary, SceneError> {
    let registered = registry
        .get(&kind)
        .ok_or_else(|| SceneError::UnknownFilterKind { kind: kind.clone() })?;

    // Unset parameters take the kind's default: this is a new filter, so there
    // is no previous value to keep.
    let params = registered.normalise(&params);
    check_bindings(registered, &params, store)?;

    let id = counter.next();

    // Empty, and registered before the first run: a caller binds these handles
    // straight out of the reply, and an actor bound to one draws nothing until
    // the filter has produced something rather than failing to resolve it.
    let mut allocated = HashMap::new();
    let mut listed = Vec::new();
    for spec in registered.outputs {
        let handle = counter.next();
        match spec.kind {
            OutputKind::Array { dtype, shape } => {
                // The **declared** shape, not `[0]`. An empty `colormap` output
                // has to describe itself as `[n, 3]` from the start, or binding
                // it to an input that takes `[n, 3]` would be refused until the
                // first run had happened — which is exactly the wait these
                // handles exist to avoid.
                let shape = shape.to_vec();
                // A run-time dtype has to be *called* something until the run
                // says otherwise. `Float32` is the placeholder, and it is only a
                // placeholder: the array is empty, and `ParamKind::accepts`
                // does not hold an empty array's dtype against it.
                let declared = dtype.unwrap_or(Dtype::Float32);
                let asset = arrays.add(DataArray::numeric(declared, shape.clone(), Vec::new()));
                store.insert(
                    handle,
                    BufferMeta {
                        name: spec.id.to_string(),
                        dtype: declared,
                        shape,
                    },
                    asset,
                );
            }
            OutputKind::Geometry => {
                // An empty triangle list, for the same reason: an actor can be
                // bound to it and placed in the scene before the filter has run,
                // and draws nothing until it has. Nothing has to be declared in
                // advance the way a shape is — a geometry input takes any mesh.
                let asset = meshes.add(empty_mesh());
                store.insert_geometry(
                    handle,
                    GeometryMeta {
                        name: spec.id.to_string(),
                        ..default()
                    },
                    asset,
                );
            }
        }
        allocated.insert(spec.id, handle);
        listed.push((spec.id.to_string(), handle));
    }

    for handle in allocated.values() {
        graph.producer.insert(*handle, id);
    }
    graph.record_reads(registry, id, registered.id, &params);

    commands.spawn((
        UniqueID(id),
        FilterKindId(registered.id),
        FilterParams(params.clone()),
        Generation::default(),
        Outputs(allocated),
    ));

    Ok(FilterSummary {
        id,
        kind,
        params,
        outputs: listed,
        // Just created or just reconfigured, so it has not run under these
        // settings yet. Whatever it last complained of describes settings that
        // no longer apply.
        problem: None,
    })
}

/// Changes a filter's parameters, merging rather than replacing.
pub fn set(
    registry: &FilterRegistry,
    graph: &mut Graph,
    store: &DataStore,
    filters: &mut FilterQuery,
    id: u64,
    params: ParamMap,
) -> Result<FilterSummary, SceneError> {
    let entity = filters
        .iter()
        .find(|(_, unique, ..)| unique.0 == id)
        .map(|(entity, ..)| entity)
        .ok_or(SceneError::NoSuchFilter(id))?;

    let (_, _, kind, current, outputs, _) = filters.get(entity).expect("just found");
    let registered = registry
        .get(kind.0)
        .ok_or_else(|| SceneError::UnknownFilterKind {
            kind: kind.0.to_string(),
        })?;

    // Built and checked whole before anything is written, so a refused command
    // leaves the filter exactly as it was rather than half-applied.
    let mut wanted = current.0.clone();
    for (key, value) in params {
        let Some(value) = registered
            .params
            .iter()
            .find(|spec| spec.id == key)
            .and_then(|spec| spec.kind.sanitise(value))
        else {
            warn!("filter: {id} has no parameter \"{key}\" of that type");
            continue;
        };
        // Clearing is a removal, not a value. The map is what everything reads
        // through, so an input that has been let go has to be *absent* from it —
        // storing a marker would make every reader learn about the marker.
        match value {
            crate::scene::registry::ParamValue::Unset => wanted.remove(&key),
            value => wanted.insert(key, value),
        };
    }
    check_bindings(registered, &wanted, store)?;

    for spec in registered.inputs() {
        let Some(handle) = crate::scene::registry::data(&wanted, spec.id) else {
            continue;
        };
        if graph.would_cycle(id, handle) {
            return Err(SceneError::FilterCycle {
                filter: id,
                input: spec.id,
            });
        }
    }

    let listed = listed_outputs(registered, outputs);
    graph.record_reads(registry, id, registered.id, &wanted);

    let (_, _, _, mut current, _, _) = filters.get_mut(entity).expect("just found");
    current.0 = wanted.clone();

    Ok(FilterSummary {
        id,
        kind: registered.id.to_string(),
        params: wanted,
        outputs: listed,
        // Just created or just reconfigured, so it has not run under these
        // settings yet. Whatever it last complained of describes settings that
        // no longer apply.
        problem: None,
    })
}

/// Removes a filter and forgets the arrays it was writing.
///
/// The handles go with it. An actor still bound to one is left bound to nothing,
/// which is the same state as binding an array that was released — it draws
/// nothing rather than drawing what the filter last happened to leave there.
pub fn remove(
    commands: &mut Commands,
    graph: &mut Graph,
    store: &mut DataStore,
    filters: &FilterQuery,
    id: u64,
) -> bool {
    let Some((entity, _, _, _, outputs, _)) = filters.iter().find(|(_, unique, ..)| unique.0 == id)
    else {
        return false;
    };

    for handle in outputs.0.values() {
        store.remove(*handle);
        graph.producer.remove(handle);
    }
    graph.reads.remove(&id);
    commands.entity(entity).despawn();
    true
}

pub fn list(registry: &FilterRegistry, filters: &FilterQuery) -> Vec<FilterSummary> {
    let mut summaries: Vec<FilterSummary> = filters
        .iter()
        .filter_map(|(_, id, kind, params, outputs, problem)| {
            let registered = registry.get(kind.0)?;
            Some(FilterSummary {
                id: id.0,
                kind: kind.0.to_string(),
                params: params.0.clone(),
                outputs: listed_outputs(registered, outputs),
                problem: problem.map(|problem| problem.0.clone()),
            })
        })
        .collect();
    // Handle order, so a listing is stable between calls.
    summaries.sort_by_key(|summary| summary.id);
    summaries
}

pub fn list_kinds(registry: &FilterRegistry) -> Vec<FilterKindSummary> {
    registry
        .iter()
        .map(|kind| FilterKindSummary {
            id: kind.id.to_string(),
            label: kind.label.to_string(),
            params: kind.params,
            outputs: kind.outputs,
        })
        .collect()
}

/// Output handles in the kind's declaration order rather than the map's, which
/// is arbitrary.
fn listed_outputs(kind: &super::FilterKind, outputs: &Outputs) -> Vec<(String, u64)> {
    kind.outputs
        .iter()
        .filter_map(|spec| Some((spec.id.to_string(), outputs.get(spec.id)?)))
        .collect()
}

/// The same gate `check_bindings` applies to an actor: every required input
/// bound, and every bound array the right type and shape for the input.
fn check_bindings(
    kind: &super::FilterKind,
    params: &ParamMap,
    store: &DataStore,
) -> Result<(), SceneError> {
    for spec in kind.inputs() {
        let required = spec.kind.is_required();
        match crate::scene::registry::data(params, spec.id) {
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
