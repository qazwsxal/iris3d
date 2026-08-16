//! Walking an element back up the graph to where it came from.
//!
//! # The problem this solves
//!
//! Subsetting used to happen inside an actor, so an actor knew the source index
//! of everything it drew and could answer "which atom is this" out of its own
//! `Remap`. Narrowing is filters now — which is right, and is what made a
//! selection shareable and computable — and the consequence is that the answer
//! moved out of the actor and into the graph.
//!
//! So a click lands on vertex 40 122 of a ribbon, and getting from there to "the
//! third atom of residue 210 of chain B" means walking backwards: through the
//! `geometry` that assembled it, the `cartoon` that generated it, the `gather`
//! that narrowed the atoms, the `subset` that chose them.
//!
//! # It costs almost nothing, because the arrays already exist
//!
//! Each output declares a [`Provenance`], and in almost every case the
//! correspondence is an array the filter already emits for another reason.
//! `cartoon` has emitted `residue_index` per vertex since it was written, so
//! that colouring a ribbon could be an ordinary `colormap`; it is also exactly
//! the vertex → residue map. `gather` is handed its own inverse as an input.
//! `subset` *is* a correspondence. So this walks and reads; it computes nothing.
//!
//! # What it cannot do, stated
//!
//! A step whose provenance is [`Provenance::Opaque`] ends the walk, and the
//! answer is "this far and no further" rather than a guess. `renumber` drops
//! entries without recording which survived; `contour` puts a vertex per crossed
//! cell; `geometry` makes one mesh out of many arrays. Each could be made
//! traceable by emitting an index array, and none is today, so the walk says so
//! instead of inventing a number.
//!
//! # Nothing calls this yet, on purpose
//!
//! It is written and tested ahead of its consumer, and the reason is a gap in
//! Bevy's mesh picking backend: it computes a `triangle_index` while raycasting
//! and then discards it — `HitData::new` takes camera, depth, position and
//! normal, and leaves `extra` as `None`. So a click currently identifies an
//! *entity* and not an element, and there is no index to walk back from.
//!
//! Getting one means casting the ray again with `MeshRayCast` and keeping the
//! triangle. That belongs with the `pick` source node, which is what will
//! actually want an element; building the walk first means the hard, quiet part
//! is settled and covered by tests rather than being written in a hurry
//! alongside the event plumbing.
#![allow(dead_code)]

use bevy::platform::collections::HashMap;

use crate::scene::DataArray;

use super::{OutputSpec, Provenance};

/// One element, in the array that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element {
    /// The handle of the array the index is *in*.
    pub handle: u64,
    pub index: u32,
}

/// Everything the walk needs about one filter, resolved for it.
///
/// A borrowed view rather than a query, so the walk is a plain function over
/// plain data and can be tested without a world. Building this from the ECS is
/// the caller's job, and is what keeps the recursion out of a Bevy system.
pub struct Step<'a> {
    /// This filter's declared outputs.
    pub outputs: &'a [OutputSpec],
    /// The handle behind each output id.
    pub produced: &'a HashMap<&'static str, u64>,
    /// The handle bound to each input id.
    pub bound: &'a HashMap<&'static str, u64>,
}

/// Which filter writes a handle, and what that filter looks like.
pub trait Graph {
    /// The filter writing `handle`, or `None` for an upload — which is where a
    /// walk ends successfully.
    fn producer_of(&self, handle: u64) -> Option<u64>;
    /// One filter's outputs and bindings.
    fn step(&self, filter: u64) -> Option<Step<'_>>;
    /// The contents of an array, for reading a correspondence out of it.
    fn array(&self, handle: u64) -> Option<&DataArray>;
}

/// How far back a walk got, and why it stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    /// Reached an array nothing produced: an upload. This is the answer a client
    /// wanted — an index into data it sent.
    Uploaded(Element),
    /// A step could not be traced further. The element is where the walk got to,
    /// which is still more useful than nothing.
    Blocked {
        at: Element,
        /// Which filter output gave up, for a message that names it.
        output: &'static str,
    },
}

impl Origin {
    /// Where the walk ended, however it ended.
    pub fn element(&self) -> Element {
        match self {
            Origin::Uploaded(element) => *element,
            Origin::Blocked { at, .. } => *at,
        }
    }
}

/// How many steps to take before giving up.
///
/// The graph is a DAG — `Graph::would_cycle` refuses anything else at bind time
/// — so this cannot be reached by a legitimate scene. It is here because a walk
/// that spins is worse than a walk that stops, and this runs in response to a
/// click.
const MAX_STEPS: usize = 64;

/// Walks an element back to the upload it came from.
///
/// Returns where it got to, and whether that was the whole way. See [`Origin`].
pub fn trace(graph: &impl Graph, from: Element) -> Origin {
    let mut at = from;

    for _ in 0..MAX_STEPS {
        let Some(filter) = graph.producer_of(at.handle) else {
            // Nothing wrote it, so it was uploaded. Done.
            return Origin::Uploaded(at);
        };
        let Some(step) = graph.step(filter) else {
            return Origin::Blocked {
                at,
                output: "unknown",
            };
        };
        // Which of this filter's outputs the element is in.
        let Some(spec) = step
            .outputs
            .iter()
            .find(|spec| step.produced.get(spec.id) == Some(&at.handle))
        else {
            return Origin::Blocked {
                at,
                output: "unknown",
            };
        };

        at = match spec.provenance {
            Provenance::Opaque => return Origin::Blocked { at, output: spec.id },
            Provenance::Identity(input) => {
                // Same position, one array upstream.
                let Some(handle) = step.bound.get(input).copied() else {
                    return Origin::Blocked { at, output: spec.id };
                };
                Element {
                    handle,
                    index: at.index,
                }
            }
            Provenance::Map { via, of } => {
                // Read the correspondence out of the array that holds it.
                let (Some(map), Some(handle)) =
                    (step.produced.get(via).copied(), step.bound.get(of).copied())
                else {
                    return Origin::Blocked { at, output: spec.id };
                };
                let Some(values) = graph.array(map).and_then(DataArray::to_u32) else {
                    return Origin::Blocked { at, output: spec.id };
                };
                let Some(index) = values.get(at.index as usize).copied() else {
                    // Past the end of the map. The filter has been re-run since
                    // the pick, most likely; better to stop than to name an
                    // element at random.
                    return Origin::Blocked { at, output: spec.id };
                };
                Element { handle, index }
            }
        };
    }

    Origin::Blocked {
        at,
        output: "too many steps",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::data::Dtype;

    /// A graph assembled by hand, so the walk is tested without a world.
    #[derive(Default)]
    struct Fake {
        /// handle -> filter that writes it.
        producer: HashMap<u64, u64>,
        /// filter -> (outputs, produced, bound)
        steps: HashMap<
            u64,
            (
                &'static [OutputSpec],
                HashMap<&'static str, u64>,
                HashMap<&'static str, u64>,
            ),
        >,
        arrays: HashMap<u64, DataArray>,
    }

    impl Graph for Fake {
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
            self.arrays.get(&handle)
        }
    }

    fn u32s(values: &[u32]) -> DataArray {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DataArray::numeric(Dtype::Uint32, vec![values.len() as u64], bytes)
    }

    const IDENTITY: &[OutputSpec] = &[OutputSpec {
        id: "colour",
        label: "colour",
        kind: super::super::OutputKind::Array {
            dtype: Some(Dtype::Float32),
            shape: &[0, 3],
        },
        provenance: Provenance::Identity("values"),
    }];

    const MAPPED: &[OutputSpec] = &[OutputSpec {
        id: "result",
        label: "result",
        kind: super::super::OutputKind::Array {
            dtype: None,
            shape: &[],
        },
        provenance: Provenance::Map {
            via: "result",
            of: "values",
        },
    }];

    const OPAQUE: &[OutputSpec] = &[OutputSpec {
        id: "geometry",
        label: "geometry",
        kind: super::super::OutputKind::Geometry,
        provenance: Provenance::Opaque,
    }];

    /// An array nobody wrote is where the walk is trying to get to.
    #[test]
    fn an_upload_is_its_own_origin() {
        let graph = Fake::default();
        let start = Element {
            handle: 1,
            index: 7,
        };
        assert_eq!(trace(&graph, start), Origin::Uploaded(start));
    }

    /// A colour map moves nothing, so the index survives the step unchanged.
    #[test]
    fn identity_keeps_the_index_and_changes_the_array() {
        let mut graph = Fake::default();
        graph.producer.insert(20, 2);
        graph.steps.insert(
            2,
            (
                IDENTITY,
                HashMap::from_iter([("colour", 20)]),
                HashMap::from_iter([("values", 1)]),
            ),
        );
        assert_eq!(
            trace(
                &graph,
                Element {
                    handle: 20,
                    index: 3
                }
            ),
            Origin::Uploaded(Element {
                handle: 1,
                index: 3
            })
        );
    }

    /// The case the whole thing exists for: a gather's own output array *is* the
    /// map back, so element 1 of the result is whatever index it holds.
    #[test]
    fn a_gather_is_walked_back_through_its_own_indices() {
        let mut graph = Fake::default();
        graph.producer.insert(30, 3);
        graph.steps.insert(
            3,
            (
                MAPPED,
                HashMap::from_iter([("result", 30)]),
                HashMap::from_iter([("values", 1)]),
            ),
        );
        // The gather kept source elements 5 and 9.
        graph.arrays.insert(30, u32s(&[5, 9]));

        assert_eq!(
            trace(
                &graph,
                Element {
                    handle: 30,
                    index: 1
                }
            ),
            Origin::Uploaded(Element {
                handle: 1,
                index: 9
            })
        );
    }

    /// Two steps, which is the shape every real chain has.
    #[test]
    fn a_chain_is_walked_the_whole_way() {
        let mut graph = Fake::default();
        // colormap(30) <- gather(20) <- upload(1)
        graph.producer.insert(30, 3);
        graph.producer.insert(20, 2);
        graph.steps.insert(
            3,
            (
                IDENTITY,
                HashMap::from_iter([("colour", 30)]),
                HashMap::from_iter([("values", 20)]),
            ),
        );
        graph.steps.insert(
            2,
            (
                MAPPED,
                HashMap::from_iter([("result", 20)]),
                HashMap::from_iter([("values", 1)]),
            ),
        );
        graph.arrays.insert(20, u32s(&[4, 8, 15]));

        assert_eq!(
            trace(
                &graph,
                Element {
                    handle: 30,
                    index: 2
                }
            ),
            Origin::Uploaded(Element {
                handle: 1,
                index: 15
            })
        );
    }

    /// An untraceable step stops the walk where it is rather than guessing, and
    /// names what gave up.
    #[test]
    fn an_opaque_step_blocks_and_says_where() {
        let mut graph = Fake::default();
        graph.producer.insert(40, 4);
        graph.steps.insert(
            4,
            (OPAQUE, HashMap::from_iter([("geometry", 40)]), HashMap::new()),
        );
        let at = Element {
            handle: 40,
            index: 2,
        };
        assert_eq!(
            trace(&graph, at),
            Origin::Blocked {
                at,
                output: "geometry"
            }
        );
    }

    /// A map shorter than the index into it means the filter re-ran since the
    /// pick. Naming an element at random would be worse than stopping.
    #[test]
    fn an_index_past_the_map_blocks() {
        let mut graph = Fake::default();
        graph.producer.insert(30, 3);
        graph.steps.insert(
            3,
            (
                MAPPED,
                HashMap::from_iter([("result", 30)]),
                HashMap::from_iter([("values", 1)]),
            ),
        );
        graph.arrays.insert(30, u32s(&[5]));

        assert!(matches!(
            trace(
                &graph,
                Element {
                    handle: 30,
                    index: 9
                }
            ),
            Origin::Blocked { .. }
        ));
    }
}
