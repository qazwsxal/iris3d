//! Turning a mask into a selection, and matching text against a name.
//!
//! [`subset`](self) is the join between the maths in [`maths`](super::maths) and
//! the narrowing in [`index`](super::index): a mask says *which* elements, and
//! this turns that into the dense index array everything downstream takes.
//!
//! # Why indices and not the mask itself
//!
//! A mask is one value per *original* element, so it says nothing about the
//! order or the count of what survives. An index array is the selection stated
//! as "these, in this order", which is what a gather reads directly and what a
//! renumber needs to rewrite connectivity. Converting once, here, also means the
//! rest of the graph never has to know which of the two spellings it was handed.
//!
//! # `match` exists because text cannot be compared arithmetically
//!
//! Residue names travel dictionary-encoded — an integer per element beside an
//! array of the distinct values ever seen — so "is this residue a water" is not
//! a numeric test. It is a lookup of `HOH` in the dictionary followed by an
//! integer comparison, and only something holding both arrays can do it.
//!
//! This is deliberately **not** a selection grammar. It matches whole names
//! against a list, with no operators, no nesting and no wildcards; anything
//! more is what `logic` is for. The app owning a query language is a thing
//! iris3d has declined more than once, and this stays on the right side of that
//! line by being a filter with two inputs rather than a parser.

use bevy::prelude::*;

use iris3d_data::array::Dtype;
use iris3d_model::{ParamKind, ParamSpec, flag, text};
use iris3d_scene::DataArray;

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

const SUBSET_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "mask",
        label: "mask (non-zero is kept)",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "invert",
        label: "invert",
        kind: ParamKind::Bool { default: false },
    },
];

const INDICES: &[OutputSpec] = &[OutputSpec {
    id: "indices",
    label: "indices",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Uint32),
        shape: &[0],
    },
    // The indices *are* the correspondence: entry i of the output names
    // element `indices[i]` of whatever the mask was over.
    provenance: Provenance::Map {
        via: "indices",
        of: "mask",
    },
}];

const MATCH_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "index",
        label: "name index per element",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "values",
        label: "distinct names",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Str],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "text",
        label: "names to match (comma separated)",
        kind: ParamKind::Text { default: "" },
    },
];

const MASK: &[OutputSpec] = &[OutputSpec {
    id: "mask",
    label: "mask",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Uint8),
        shape: &[0],
    },
    // One value per element of the index array, in order. Not of `values`:
    // that is the dictionary, and it has a different length entirely.
    provenance: Provenance::Identity("index"),
}];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "subset",
        label: "subset",
        params: SUBSET_PARAMS,
        outputs: INDICES,
        run: Some(run_subset),
    });
    registry.register(FilterKind {
        id: "match",
        label: "match names",
        params: MATCH_PARAMS,
        outputs: MASK,
        run: Some(run_match),
    });
}

fn run_subset(request: &Request) -> Outcome {
    let Some(mask) = request.input("mask") else {
        return Outcome::refused("has nothing bound to \"mask\"");
    };
    let invert = flag(&request.params, "invert", false);
    let values = mask.to_f32();

    let indices: Vec<u32> = values
        .iter()
        .enumerate()
        .filter(|(_, value)| (**value != 0.0) != invert)
        .map(|(index, _)| index as u32)
        .collect();

    // An empty selection is a *fact*, not a failure — "no waters in this
    // structure" is a true answer and the graph downstream handles it. But it
    // is also what a mis-wired predicate looks like, so it is said out loud
    // while still producing the empty array.
    let empty = indices.is_empty();
    let bytes: Vec<u8> = indices
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();

    let mut products = Products::new();
    products.insert(
        "indices",
        DataArray::numeric(Dtype::Uint32, vec![indices.len() as u64], bytes).into(),
    );
    let outcome: Outcome = products.into();
    match empty {
        true => outcome.but(format!("selected none of its {} elements", values.len())),
        false => outcome,
    }
}

fn run_match(request: &Request) -> Outcome {
    let (Some(index), Some(values)) = (request.input("index"), request.input("values")) else {
        return Outcome::refused("needs both an index array and the distinct names");
    };
    if values.dtype != Dtype::Str {
        return Outcome::refused("was given something other than text as its names");
    }
    let Some(index) = index.to_u32() else {
        return Outcome::refused("was given name indices that are not an integer type");
    };

    let wanted: Vec<&str> = text(&request.params, "text", "")
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect();
    if wanted.is_empty() {
        return Outcome::refused("has no names to match: type one into \"names to match\"");
    }

    // Resolved against the dictionary once rather than per element. The names
    // that matched nothing are worth reporting: a typo — `HOH` against a file
    // that spells it `WAT` — otherwise looks exactly like a structure with no
    // waters in it.
    let dictionary = &values.strings;
    let hits: Vec<u32> = dictionary
        .iter()
        .enumerate()
        .filter(|(_, name)| wanted.contains(&name.as_str()))
        .map(|(slot, _)| slot as u32)
        .collect();
    let missed: Vec<&str> = wanted
        .iter()
        .filter(|name| !dictionary.iter().any(|held| held == *name))
        .copied()
        .collect();

    let mask: Vec<u8> = index
        .iter()
        .map(|slot| u8::from(hits.contains(slot)))
        .collect();

    let mut products = Products::new();
    products.insert(
        "mask",
        DataArray::numeric(Dtype::Uint8, vec![mask.len() as u64], mask).into(),
    );
    let outcome: Outcome = products.into();
    match missed.is_empty() {
        true => outcome,
        false => outcome.but(format!(
            "found no {} among the {} names present",
            missed.join(", "),
            dictionary.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;
    use iris3d_model::{ParamMap, ParamValue};

    fn numbers(values: &[f32]) -> DataArray {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DataArray::numeric(Dtype::Float32, vec![values.len() as u64], bytes)
    }

    fn indices(values: &[u32]) -> DataArray {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DataArray::numeric(Dtype::Uint32, vec![values.len() as u64], bytes)
    }

    fn strings(values: &[&str]) -> DataArray {
        DataArray {
            dtype: Dtype::Str,
            shape: vec![values.len() as u64],
            data: Vec::new(),
            strings: values.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn read_u32(outcome: &Outcome, id: &str) -> Vec<u32> {
        outcome.products[id]
            .array()
            .expect("an array")
            .to_u32()
            .expect("integers")
    }

    #[test]
    fn a_mask_becomes_the_indices_it_keeps() {
        let mut inputs = HashMap::new();
        inputs.insert("mask", numbers(&[0.0, 1.0, 0.0, 1.0]));
        let out = run_subset(&Request {
            params: ParamMap::new(),
            inputs,
        });
        assert_eq!(read_u32(&out, "indices"), vec![1, 3]);
    }

    #[test]
    fn inverting_keeps_the_others() {
        let mut inputs = HashMap::new();
        inputs.insert("mask", numbers(&[0.0, 1.0, 0.0, 1.0]));
        let mut params = ParamMap::new();
        params.insert("invert".into(), ParamValue::Bool(true));
        let out = run_subset(&Request { params, inputs });
        assert_eq!(read_u32(&out, "indices"), vec![0, 2]);
    }

    /// Selecting nothing is a real answer, so the array is still produced — but
    /// it is the same picture a mis-wired predicate gives, so it is also said.
    #[test]
    fn selecting_nothing_still_produces_an_array_and_says_so() {
        let mut inputs = HashMap::new();
        inputs.insert("mask", numbers(&[0.0, 0.0]));
        let out = run_subset(&Request {
            params: ParamMap::new(),
            inputs,
        });
        assert_eq!(read_u32(&out, "indices"), Vec::<u32>::new());
        assert!(out.problem.is_some(), "an empty selection should be noted");
    }

    #[test]
    fn matching_a_name_marks_every_element_pointing_at_it() {
        let mut inputs = HashMap::new();
        inputs.insert("index", indices(&[0, 1, 2, 1]));
        inputs.insert("values", strings(&["ALA", "HOH", "GLY"]));
        let mut params = ParamMap::new();
        params.insert("text".into(), ParamValue::Text("HOH".into()));
        let out = run_match(&Request { params, inputs });
        assert_eq!(
            out.products["mask"].array().expect("array").data,
            vec![0, 1, 0, 1]
        );
    }

    /// A name that is not in the file is almost always a typo, and it produces
    /// exactly the same empty mask as a correct name with no matches.
    #[test]
    fn a_name_that_is_not_there_is_reported() {
        let mut inputs = HashMap::new();
        inputs.insert("index", indices(&[0, 1]));
        inputs.insert("values", strings(&["ALA", "WAT"]));
        let mut params = ParamMap::new();
        params.insert("text".into(), ParamValue::Text("HOH".into()));
        let out = run_match(&Request { params, inputs });
        assert!(out.problem.expect("noted").contains("HOH"));
    }
}
