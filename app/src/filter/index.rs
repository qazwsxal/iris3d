//! Reading one array through another: the hierarchy join, and the narrowing.
//!
//! [`gather`](self) is the keystone of the whole maths set, and it does two jobs
//! that turn out to be the same job.
//!
//! # Joining the levels of a hierarchy
//!
//! Molecular data travels as dense index arrays plus side arrays keyed on them:
//! `residue_index` per atom, `residue_chain_index` per residue, `chain_id` per
//! chain. The format is good and the levels never met, because nothing could
//! evaluate `residue_chain_index[residue_index]`.
//!
//! That gap is why **a cartoon could not be coloured by chain** — the single
//! most ordinary thing anyone does to a multimer. `cartoon` emits
//! `residue_index` *per vertex*, so a chain colour is one gather away and was
//! unreachable without it. `filter/cartoon.rs` has said so in a comment since
//! the day it was written: "or through a gather for anything else keyed on
//! residue".
//!
//! # Applying a subset
//!
//! `positions[indices]` is the same operation. This is what lets an actor stay a
//! dumb consumer: a subset is applied *before* the actor by narrowing the arrays
//! it binds, rather than by the actor deciding for itself what to draw. Deciding
//! what to draw is not an actor's job, and three kinds used to do it anyway.
//!
//! # Three operations, because three kinds of array narrow differently
//!
//! - **Per-element data** — [`gather`]. One element out per index in.
//! - **Connectivity** — [`renumber`]. A bond or a triangle survives only if
//!   every endpoint did, and the survivors are renumbered into the new, denser
//!   space. Keeping one with a dropped end would mean inventing a vertex.
//! - **Dense hierarchy indices** — [`reindex`]. Narrowing atoms leaves
//!   `residue_index` full of gaps, and everything downstream assumes it is
//!   dense. This re-densifies, and emits the `kept` array that narrows the
//!   residue-keyed side arrays in turn.
//!
//! The renumbering rule is VTK's extract-selection rule and is not reinvented
//! here: [`Remap`] already implemented it for the actor-side subset that this
//! replaces.

use bevy::prelude::*;

use crate::scene::DataArray;
use crate::scene::data::Dtype;
use crate::scene::registry::{ParamKind, ParamSpec};
use crate::scene::subset::Remap;

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

/// An index array: integers, one per element.
const INDICES_IN: ParamKind = ParamKind::Array {
    dtypes: &[],
    shape: &[0],
    required: true,
    structural: true,
};

const GATHER_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "values",
        label: "values",
        kind: ParamKind::Array {
            // Any type and any shape: what comes back is what went in, one
            // element at a time, so nothing here needs to understand it.
            dtypes: &[],
            shape: &[],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "indices",
        label: "indices",
        kind: INDICES_IN,
    },
];

const GATHERED: &[OutputSpec] = &[OutputSpec {
    id: "result",
    label: "result",
    kind: OutputKind::Array {
        // Decided by the run: elements of whatever was bound. Declaring
        // `Float32` here would mean a gathered `elements` array could not be
        // bound back into `ball-and-stick`, which takes `Uint8` only.
        dtype: None,
        shape: &[],
    },
    // The whole point of the filter, stated: element i out is element
    // `indices[i]` of `values`. This is the link a pick walks back along.
    provenance: Provenance::Map {
        via: "result",
        of: "values",
    },
}];

const RENUMBER_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "connectivity",
        label: "bonds or triangles",
        kind: ParamKind::Array {
            dtypes: &[],
            // Two-dimensional, any width: [n, 2] for bonds, [n, 3] for
            // triangles. One filter serves both because the rule is the same.
            shape: &[0, 0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "indices",
        label: "elements kept",
        kind: INDICES_IN,
    },
];

const RENUMBERED: &[OutputSpec] = &[OutputSpec {
    id: "connectivity",
    label: "bonds or triangles",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Uint32),
        shape: &[0, 0],
    },
    // Entries are dropped, not reordered, and nothing records which survived —
    // so entry i out is not entry i in, and there is no array saying what it
    // is. Honest rather than convenient.
    provenance: Provenance::Opaque,
}];

const REINDEX_PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "values",
    label: "sparse indices",
    kind: INDICES_IN,
}];

const REINDEXED: &[OutputSpec] = &[
    OutputSpec {
        id: "result",
        label: "dense indices",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[0],
        },
        // Renumbered in place: one value out per value in, same order. Only
        // what the numbers *say* changed.
        provenance: Provenance::Identity("values"),
    },
    // The other half, and the one easy to leave out: re-densifying
    // `residue_index` is useless on its own, because every array keyed on the
    // old numbering — `residue_name_index`, `residue_sse`, `residue_snfg` — now
    // points at the wrong rows. This says which rows survived, so a `gather`
    // narrows them to match.
    OutputSpec {
        id: "kept",
        label: "values that survived",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[0],
        },
        // A list of the old numbers, so it indexes the space `values` pointed
        // *into* — the residues — not `values` itself, which is per atom. There
        // is no input here naming that space, so it cannot be stated.
        provenance: Provenance::Opaque,
    },
];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "gather",
        label: "gather",
        params: GATHER_PARAMS,
        outputs: GATHERED,
        run: run_gather,
    });
    registry.register(FilterKind {
        id: "renumber",
        label: "renumber",
        params: RENUMBER_PARAMS,
        outputs: RENUMBERED,
        run: run_renumber,
    });
    registry.register(FilterKind {
        id: "reindex",
        label: "reindex",
        params: REINDEX_PARAMS,
        outputs: REINDEXED,
        run: run_reindex,
    });
}

/// Copies whole elements out of `array`, by index, keeping the element type.
///
/// Works on raw bytes rather than on decoded numbers, which is what makes the
/// dtype survive: gathering a `Uint8` element array gives back `Uint8`, and a
/// `[n, 3]` position array gives back `[m, 3]`.
fn take(array: &DataArray, indices: &[u32]) -> DataArray {
    let stride = array.dtype.size() as usize * array.components().max(1) as usize;
    let mut data = Vec::with_capacity(indices.len() * stride);
    let mut strings = Vec::new();

    for index in indices {
        let start = *index as usize * stride;
        // Text carries no bytes at all — its elements live in `strings` — so it
        // is gathered there instead. See `Dtype::Str`.
        if array.dtype == Dtype::Str {
            if let Some(value) = array.strings.get(*index as usize) {
                strings.push(value.clone());
            }
            continue;
        }
        data.extend_from_slice(&array.data[start..start + stride]);
    }

    // The leading axis becomes the number gathered; the rest is unchanged,
    // because a gather takes whole elements.
    let mut shape = array.shape.clone();
    if shape.is_empty() {
        shape.push(indices.len() as u64);
    } else {
        shape[0] = indices.len() as u64;
    }
    DataArray {
        dtype: array.dtype,
        shape,
        data,
        strings,
    }
}

fn run_gather(request: &Request) -> Outcome {
    let (Some(values), Some(indices)) = (request.input("values"), request.input("indices")) else {
        return Outcome::refused("needs both \"values\" and \"indices\"");
    };
    let Some(indices) = indices.to_u32() else {
        return Outcome::refused("was given indices that are not an integer type");
    };

    let count = values.count() as usize;
    if let Some(past) = indices.iter().find(|index| **index as usize >= count) {
        return Outcome::refused(format!(
            "has index {past}, past the {count} values it reads from"
        ));
    }

    let mut products = Products::new();
    products.insert("result", take(values, &indices).into());
    products.into()
}

fn run_renumber(request: &Request) -> Outcome {
    let (Some(connectivity), Some(indices)) =
        (request.input("connectivity"), request.input("indices"))
    else {
        return Outcome::refused("needs both \"connectivity\" and the elements kept");
    };
    let (Some(entries), Some(kept)) = (connectivity.to_u32(), indices.to_u32()) else {
        return Outcome::refused("was given something that is not an integer type");
    };
    let width = connectivity.components().max(1) as usize;
    if width < 2 {
        return Outcome::refused(format!(
            "was given {width}-wide connectivity, and an entry joins at least two elements"
        ));
    }

    // The widest index decides how large the original space was. Taken from the
    // data rather than asked for: a caller who had to state it could state it
    // wrongly, and the arrays already know.
    let highest = entries.iter().copied().max().unwrap_or(0) as usize;
    let source = highest.max(kept.iter().copied().max().unwrap_or(0) as usize) + 1;
    let remap = Remap::new(&kept, source);

    let mut survived: Vec<u32> = Vec::new();
    for entry in entries.chunks_exact(width) {
        // Every end has to survive. This is VTK's extract-selection rule, and
        // `Remap::cell` is the same code the actor-side subset used.
        if let Some(renumbered) = remap.cell(entry) {
            survived.extend_from_slice(&renumbered);
        }
    }

    let rows = survived.len() / width;
    let bytes: Vec<u8> = survived
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    let mut products = Products::new();
    products.insert(
        "connectivity",
        DataArray::numeric(Dtype::Uint32, vec![rows as u64, width as u64], bytes).into(),
    );
    let outcome: Outcome = products.into();
    match rows {
        0 => outcome.but(format!(
            "kept none of its {} entries: every one had an end that was cut",
            entries.len() / width
        )),
        _ => outcome,
    }
}

fn run_reindex(request: &Request) -> Outcome {
    let Some(values) = request.input("values") else {
        return Outcome::refused("has nothing bound to \"values\"");
    };
    let Some(values) = values.to_u32() else {
        return Outcome::refused("was given values that are not an integer type");
    };

    // The distinct values still present, in order. That order *is* the new
    // numbering, and the list of them is exactly the `kept` array a downstream
    // gather needs.
    let mut kept: Vec<u32> = values.clone();
    kept.sort_unstable();
    kept.dedup();

    let dense: Vec<u32> = values
        .iter()
        .map(|value| {
            kept.binary_search(value)
                .map(|slot| slot as u32)
                .unwrap_or(0)
        })
        .collect();

    let mut products = Products::new();
    products.insert(
        "result",
        DataArray::numeric(
            Dtype::Uint32,
            vec![dense.len() as u64],
            dense.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .into(),
    );
    products.insert(
        "kept",
        DataArray::numeric(
            Dtype::Uint32,
            vec![kept.len() as u64],
            kept.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .into(),
    );
    products.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::registry::ParamMap;
    use bevy::platform::collections::HashMap;

    fn u32s(values: &[u32]) -> DataArray {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DataArray::numeric(Dtype::Uint32, vec![values.len() as u64], bytes)
    }

    fn request(inputs: Vec<(&'static str, DataArray)>) -> Request {
        let mut bound = HashMap::new();
        for (id, array) in inputs {
            bound.insert(id, array);
        }
        Request {
            params: ParamMap::new(),
            inputs: bound,
        }
    }

    fn out_u32(outcome: &Outcome, id: &str) -> Vec<u32> {
        outcome.products[id]
            .array()
            .expect("array")
            .to_u32()
            .expect("integers")
    }

    /// The hierarchy join: chain per residue, read through residue per atom,
    /// gives chain per atom. This is colour-by-chain in one node.
    #[test]
    fn gathering_joins_two_levels_of_a_hierarchy() {
        let out = run_gather(&request(vec![
            // Residue 0 and 1 are in chain 0; residue 2 is in chain 1.
            ("values", u32s(&[0, 0, 1])),
            // Four atoms, in residues 0, 0, 1, 2.
            ("indices", u32s(&[0, 0, 1, 2])),
        ]));
        assert_eq!(out_u32(&out, "result"), vec![0, 0, 0, 1]);
    }

    /// The dtype has to survive, or a gathered `elements` array cannot be bound
    /// back into `ball-and-stick`, which takes `Uint8` only.
    #[test]
    fn gathering_keeps_the_element_type() {
        let elements = DataArray::numeric(Dtype::Uint8, vec![3], vec![6, 7, 8]);
        let out = run_gather(&request(vec![
            ("values", elements),
            ("indices", u32s(&[2, 0])),
        ]));
        let array = out.products["result"].array().expect("array");
        assert_eq!(array.dtype, Dtype::Uint8);
        assert_eq!(array.data, vec![8, 6]);
    }

    /// And so does the shape, or gathered positions cannot be bound as
    /// positions.
    #[test]
    fn gathering_keeps_the_components() {
        let positions = DataArray::numeric(
            Dtype::Float32,
            vec![3, 3],
            (0..9).flat_map(|v| (v as f32).to_le_bytes()).collect(),
        );
        let out = run_gather(&request(vec![
            ("values", positions),
            ("indices", u32s(&[2, 0])),
        ]));
        let array = out.products["result"].array().expect("array");
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(array.to_f32(), vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0]);
    }

    /// Reading past the end is a wiring mistake — two arrays over different
    /// things — and it says which index and how many there were.
    #[test]
    fn an_index_past_the_end_is_refused_with_both_numbers() {
        let out = run_gather(&request(vec![
            ("values", u32s(&[1, 2])),
            ("indices", u32s(&[0, 5])),
        ]));
        let problem = out.problem.expect("refused");
        assert!(problem.contains('5') && problem.contains('2'), "{problem}");
    }

    /// A bond survives only if both of its atoms did, and is renumbered into
    /// the space the survivors now occupy.
    #[test]
    fn renumbering_drops_an_entry_with_a_cut_end() {
        let bonds = DataArray::numeric(
            Dtype::Uint32,
            vec![3, 2],
            [0u32, 1, 1, 2, 2, 3]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        // Keep atoms 1 and 2, so only the 1-2 bond can survive, as 0-1.
        let out = run_renumber(&request(vec![
            ("connectivity", bonds),
            ("indices", u32s(&[1, 2])),
        ]));
        assert_eq!(out_u32(&out, "connectivity"), vec![0, 1]);
    }

    /// Narrowing atoms leaves the residue index full of gaps. Everything
    /// downstream assumes it is dense, so this closes them — and says which
    /// rows survived, so the residue-keyed arrays can be narrowed to match.
    #[test]
    fn reindexing_closes_the_gaps_and_reports_what_survived() {
        let out = run_reindex(&request(vec![("values", u32s(&[3, 3, 7, 9, 7]))]));
        assert_eq!(out_u32(&out, "result"), vec![0, 0, 1, 2, 1]);
        assert_eq!(out_u32(&out, "kept"), vec![3, 7, 9]);
    }
}
