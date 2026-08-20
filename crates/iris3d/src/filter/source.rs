//! `pick` as a source: a filter-shaped entity whose output changes on a click
//! rather than on its inputs.
//!
//! # Why this lives in `FilterRegistry` rather than a sibling registry
//!
//! A `SourceKind` sibling to [`FilterKind`] was weighed and rejected before
//! this was written. The registry, `AddFilter`/`SetFilter`/`ListFilters`, the
//! node canvas and the parameter panel are already generic over any kind with
//! `params`/`outputs` in this shape — none of them call `run`, only [`start`](super::start)
//! does — so a source registers here for free and a sibling registry would
//! have bought type purity at the cost of a second, permanently-parallel RPC
//! family for what is, today, one kind.
//!
//! # What ships here, and what does not
//!
//! **`pick` only.** It has no parameters: every click *replaces* the
//! selection, walked all the way back to the array a client uploaded. Two
//! things named in the plan are deliberately not built yet:
//!
//! - **`hover`.** Nothing in [`viewport::pick`](crate::viewport::pick) raises
//!   a hover event to write from; adding one is its own piece of work.
//! - **Choosing which upstream array the indices are expressed in.** Ending
//!   the walk at [`Origin::Uploaded`] is the common case — a client's own
//!   upload is usually the level a mask is wanted at — but a scene with
//!   several hierarchy levels between the pick and the upload cannot ask for
//!   an intermediate one. That needs a parameter and a version of
//!   [`trace`](provenance::trace) that stops early, and is follow-up work
//!   rather than a silent simplification.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::scene::Picked;
use crate::scene::{DataArray, DataStore};
use iris3d_core::counter::UniqueID;
use iris3d_data::array::{BufferMeta, Dtype};
use iris3d_model::Bindings;

use super::provenance::{self, Element, Origin, Step};
use super::{
    FilterKind, FilterKindId, FilterRegistry, OutputKind, OutputSpec, Outputs, Provenance,
};

const OUTPUTS: &[OutputSpec] = &[
    OutputSpec {
        id: "element",
        label: "picked element",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[1],
        },
        // Names one index into an upload; nothing consumes a pick's own
        // output as something to walk further back through.
        provenance: Provenance::Opaque,
    },
    OutputSpec {
        id: "mask",
        label: "picked mask",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint8),
            shape: &[0],
        },
        provenance: Provenance::Opaque,
    },
];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "pick",
        label: "pick",
        params: &[],
        outputs: OUTPUTS,
        // A source: nothing for `start` to schedule. See the module doc.
        run: None,
    });
}

/// Turns a [`Picked`] event into a write straight into every `pick` source's
/// [`Outputs`], bypassing `run`/`collect` entirely.
///
/// Rewrites the existing assets in place, exactly as `collect` does for a
/// computed filter — which is what raises the `AssetEvent::Modified` that
/// `mark_stale` and `draw::mark_dirty` already watch. Nothing downstream has
/// to learn that this write came from a click rather than a run.
pub fn write_picks(
    mut picks: MessageReader<Picked>,
    registry: Res<FilterRegistry>,
    filters: Query<(&UniqueID, &FilterKindId, &Outputs, Option<&Bindings>), With<FilterKindId>>,
    actor_bindings: Query<&Bindings>,
    mut arrays: ResMut<Assets<DataArray>>,
    mut store: ResMut<DataStore>,
) {
    // Nothing bound "pick" is listening, so the walk is not worth building.
    let mut sinks = filters
        .iter()
        .filter(|(_, kind, ..)| kind.0 == "pick")
        .peekable();
    if sinks.peek().is_none() {
        return;
    }

    for pick in picks.read() {
        let Some(element) = pick.element else {
            continue;
        };
        let Ok(bound) = actor_bindings.get(pick.actor) else {
            continue;
        };
        let Some(geometry) = bound.get("geometry") else {
            continue;
        };

        let graph = WorldGraph::build(&registry, &filters, &arrays, &store);
        let origin = provenance::trace(
            &graph,
            Element {
                handle: geometry,
                index: element,
            },
        );
        let Origin::Uploaded(source) = origin else {
            // Blocked partway back — `geometry` is Opaque, or a filter
            // upstream has not been taught its own provenance yet. Nothing
            // to write; a client asking why sees no selection rather than a
            // wrong one.
            continue;
        };

        let Some(total) = store.array(source.handle).map(|held| held.meta.shape[0]) else {
            continue;
        };

        for (_, _, outputs, _) in filters.iter().filter(|(_, kind, ..)| kind.0 == "pick") {
            write_element(&mut arrays, &mut store, outputs, source.index);
            write_mask(&mut arrays, &mut store, outputs, source.index, total);
        }
    }
}

fn write_element(
    arrays: &mut Assets<DataArray>,
    store: &mut DataStore,
    outputs: &Outputs,
    index: u32,
) {
    let Some(handle) = outputs.get("element") else {
        return;
    };
    let Some(held) = store.array(handle) else {
        return;
    };
    let asset = held.handle.clone();
    let Some(mut existing) = arrays.get_mut(&asset) else {
        return;
    };
    *existing = DataArray::numeric(Dtype::Uint32, vec![1], index.to_le_bytes().to_vec());
    store.insert(
        handle,
        BufferMeta {
            name: "element".to_string(),
            dtype: Dtype::Uint32,
            shape: vec![1],
        },
        asset,
    );
}

fn write_mask(
    arrays: &mut Assets<DataArray>,
    store: &mut DataStore,
    outputs: &Outputs,
    index: u32,
    total: u64,
) {
    let Some(handle) = outputs.get("mask") else {
        return;
    };
    let Some(held) = store.array(handle) else {
        return;
    };
    let asset = held.handle.clone();
    let mut mask = vec![0u8; total as usize];
    if let Some(slot) = mask.get_mut(index as usize) {
        *slot = 1;
    }
    let Some(mut existing) = arrays.get_mut(&asset) else {
        return;
    };
    *existing = DataArray::numeric(Dtype::Uint8, vec![total], mask);
    store.insert(
        handle,
        BufferMeta {
            name: "mask".to_string(),
            dtype: Dtype::Uint8,
            shape: vec![total],
        },
        asset,
    );
}

/// [`provenance::Graph`] over the live world, assembled fresh per pick.
///
/// Cheap enough to rebuild on every click — a scene has tens of filters, not
/// thousands — and rebuilding avoids keeping a second copy of the filter
/// graph as a resource in step with `scene::mod`'s own, command-scoped one.
struct WorldGraph<'a> {
    steps: super::Steps,
    producer: HashMap<u64, u64>,
    arrays: &'a Assets<DataArray>,
    store: &'a DataStore,
}

impl<'a> WorldGraph<'a> {
    fn build(
        registry: &FilterRegistry,
        filters: &Query<
            (&UniqueID, &FilterKindId, &Outputs, Option<&Bindings>),
            With<FilterKindId>,
        >,
        arrays: &'a Assets<DataArray>,
        store: &'a DataStore,
    ) -> Self {
        let mut steps = HashMap::new();
        let mut producer = HashMap::new();
        for (id, kind, outputs, bound) in filters.iter() {
            for handle in outputs.0.values() {
                producer.insert(*handle, id.0);
            }
            if let Some(registered) = registry.get(kind.0) {
                let bound = bound.map(|b| b.0.clone()).unwrap_or_default();
                steps.insert(id.0, (registered.outputs, outputs.0.clone(), bound));
            }
        }
        Self {
            steps,
            producer,
            arrays,
            store,
        }
    }
}

impl<'a> provenance::Graph for WorldGraph<'a> {
    fn producer_of(&self, handle: u64) -> Option<u64> {
        self.producer.get(&handle).copied()
    }

    fn step(&self, filter: u64) -> Option<Step<'_>> {
        let (outputs, produced, bound) = self.steps.get(&filter)?;
        Some(Step {
            outputs,
            produced,
            bound,
        })
    }

    fn array(&self, handle: u64) -> Option<&DataArray> {
        let held = self.store.array(handle)?;
        self.arrays.get(&held.handle)
    }
}
