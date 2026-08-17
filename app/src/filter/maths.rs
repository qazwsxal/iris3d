//! Arithmetic, comparison and boolean logic over arrays.
//!
//! Three filters, one shape: an array in, an array out, one element at a time.
//! Together with [`select`](super::select) and [`index`](super::index) they are
//! what lets a *selection* be expressed in the graph rather than computed by a
//! client and uploaded — "chain 1, not water, B-factor above 40" is a handful of
//! nodes, and every one of them is draggable.
//!
//! # The graph is the expression
//!
//! There is deliberately no expression language. A text field holding
//! `chain_index == 1 and not hetero` would need a parser, a type checker, a
//! sandbox and a maintenance commitment, and it would be a second way of saying
//! what a wire already says. Wiring `compare` into `logic` is more verbose and
//! has none of that behind it — and it is *inspectable*: every intermediate is
//! an ordinary array with a handle, so a selection that is wrong can be looked
//! at half-built.
//!
//! # Scalars only, and why the restriction is worth having
//!
//! Every input here takes a one-component array. A comparison against a
//! three-component array has no obvious meaning — is `positions > 0` a test per
//! component, or on the magnitude, or on x? — and each answer is defensible,
//! which is exactly when a filter should refuse rather than choose. Magnitude is
//! what [`colormap`](super::colormap) does, and it is right there because a
//! colour needs *some* number; a predicate does not have to invent one.
//!
//! Reducing a vector to a scalar is therefore a separate step, and when one is
//! wanted it belongs here as a `magnitude` filter rather than as a hidden rule
//! inside these.
//!
//! # Masks are `Uint8`, non-zero to keep
//!
//! There is no boolean dtype, and none is needed:
//! [`Subset`](crate::scene::subset) has always read a mask as "non-zero to
//! keep", so the convention exists and is followed rather than extended.

use bevy::prelude::*;

use crate::scene::DataArray;
use crate::scene::data::Dtype;
use crate::scene::registry::{ParamKind, ParamSpec, float, text};

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

/// A scalar array, as every input here takes one.
const SCALAR: ParamKind = ParamKind::Array {
    dtypes: &[],
    shape: &[0],
    required: true,
    structural: true,
};

/// The optional right-hand side. Unbound means the `value` parameter is used.
const SCALAR_OPTIONAL: ParamKind = ParamKind::Array {
    dtypes: &[],
    shape: &[0],
    required: false,
    structural: true,
};

/// A literal, for the common case of testing against a number rather than
/// against another array.
///
/// Wide open on range because it is compared against whatever the data holds,
/// and a residue index, a B-factor and a coordinate share no scale.
const VALUE: ParamKind = ParamKind::Float {
    default: 0.0,
    min: -1.0e30,
    max: 1.0e30,
    logarithmic: false,
};

const COMPARISONS: &[&str] = &["==", "!=", "<", "<=", ">", ">="];
const OPERATIONS: &[&str] = &["+", "-", "*", "/", "min", "max"];
const CONNECTIVES: &[&str] = &["and", "or", "xor", "not"];

const COMPARE_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "a",
        label: "values",
        kind: SCALAR,
    },
    ParamSpec {
        id: "b",
        label: "against (or use the value below)",
        kind: SCALAR_OPTIONAL,
    },
    ParamSpec {
        id: "op",
        label: "test",
        kind: ParamKind::Choice {
            options: COMPARISONS,
            default: "==",
        },
    },
    ParamSpec {
        id: "value",
        label: "value",
        kind: VALUE,
    },
];

const ARITHMETIC_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "a",
        label: "values",
        kind: SCALAR,
    },
    ParamSpec {
        id: "b",
        label: "and (or use the value below)",
        kind: SCALAR_OPTIONAL,
    },
    ParamSpec {
        id: "op",
        label: "operation",
        kind: ParamKind::Choice {
            options: OPERATIONS,
            default: "+",
        },
    },
    ParamSpec {
        id: "value",
        label: "value",
        kind: VALUE,
    },
];

const LOGIC_PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "a",
        label: "mask",
        kind: SCALAR,
    },
    ParamSpec {
        id: "b",
        label: "and mask (not needed by \"not\")",
        kind: SCALAR_OPTIONAL,
    },
    ParamSpec {
        id: "op",
        label: "connective",
        kind: ParamKind::Choice {
            options: CONNECTIVES,
            default: "and",
        },
    },
];

const MASK: &[OutputSpec] = &[OutputSpec {
    id: "mask",
    label: "mask",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Uint8),
        shape: &[0],
    },
    // One value out per value in, in the same order.
    provenance: Provenance::Identity("a"),
}];

const RESULT: &[OutputSpec] = &[OutputSpec {
    id: "result",
    label: "result",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Float32),
        shape: &[0],
    },
    provenance: Provenance::Identity("a"),
}];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "compare",
        label: "compare",
        params: COMPARE_PARAMS,
        outputs: MASK,
        run: Some(run_compare),
    });
    registry.register(FilterKind {
        id: "arithmetic",
        label: "arithmetic",
        params: ARITHMETIC_PARAMS,
        outputs: RESULT,
        run: Some(run_arithmetic),
    });
    registry.register(FilterKind {
        id: "logic",
        label: "logic",
        params: LOGIC_PARAMS,
        outputs: MASK,
        run: Some(run_logic),
    });
}

/// The two sides of an elementwise filter, already length-matched.
///
/// `b` is `None` only for `not`, which is the one operation here that takes a
/// single argument.
struct Pair {
    a: Vec<f32>,
    b: Option<Vec<f32>>,
}

/// Reads `a` and `b`, applying the broadcast rule and the scalar restriction.
///
/// **The broadcast rule:** equal lengths, or a right-hand side of one element
/// that applies to every element on the left. Anything else is refused *naming
/// both lengths*, because that is the whole diagnosis — two arrays over
/// different things is the routine mistake once there is arithmetic in a graph,
/// and "produced nothing" would be an unusable way to report it.
fn read(request: &Request, literal: Option<f32>) -> Result<Pair, String> {
    let Some(left) = request.input("a") else {
        return Err("has nothing bound to its first input".into());
    };
    if left.components() > 1 {
        return Err(format!(
            "was given a {}-component array, and this filter takes scalars: \
             reduce it to one value per element first",
            left.components()
        ));
    }
    let a = left.to_f32();

    let right = match request.input("b") {
        None => return Ok(Pair { a, b: literal.map(|value| vec![value]) }),
        Some(right) => right,
    };
    if right.components() > 1 {
        return Err(format!(
            "was given a {}-component array on its second input, and this \
             filter takes scalars",
            right.components()
        ));
    }
    let b = right.to_f32();
    if b.len() != a.len() && b.len() != 1 {
        return Err(format!(
            "was given {} values on one side and {} on the other: they must \
             match, or one must be a single value",
            a.len(),
            b.len()
        ));
    }
    Ok(Pair { a, b: Some(b) })
}

/// `b` at this element, given the broadcast rule already checked in [`read`].
fn at(b: &[f32], index: usize) -> f32 {
    match b.len() {
        1 => b[0],
        _ => b[index],
    }
}

fn mask(values: Vec<u8>) -> Products {
    let mut products = Products::new();
    products.insert(
        "mask",
        DataArray::numeric(Dtype::Uint8, vec![values.len() as u64], values).into(),
    );
    products
}

fn run_compare(request: &Request) -> Outcome {
    let literal = float(&request.params, "value", 0.0);
    let pair = match read(request, Some(literal)) {
        Ok(pair) => pair,
        Err(why) => return Outcome::refused(why),
    };
    let Some(b) = pair.b else {
        return Outcome::refused("has nothing to compare against");
    };
    let op = text(&request.params, "op", "==").to_string();

    let values = pair
        .a
        .iter()
        .enumerate()
        .map(|(index, left)| {
            let right = at(&b, index);
            let kept = match op.as_str() {
                "==" => left == &right,
                "!=" => left != &right,
                "<" => *left < right,
                "<=" => *left <= right,
                ">" => *left > right,
                ">=" => *left >= right,
                _ => false,
            };
            u8::from(kept)
        })
        .collect();
    mask(values).into()
}

fn run_arithmetic(request: &Request) -> Outcome {
    let literal = float(&request.params, "value", 0.0);
    let pair = match read(request, Some(literal)) {
        Ok(pair) => pair,
        Err(why) => return Outcome::refused(why),
    };
    let Some(b) = pair.b else {
        return Outcome::refused("has nothing to combine with");
    };
    let op = text(&request.params, "op", "+").to_string();

    let mut bytes = Vec::with_capacity(pair.a.len() * 4);
    for (index, left) in pair.a.iter().enumerate() {
        let right = at(&b, index);
        let value = match op.as_str() {
            "+" => left + right,
            "-" => left - right,
            "*" => left * right,
            // Division by zero gives an infinity rather than refusing the whole
            // run. One bad element should not blank an array of a million, and
            // a colour map already handles a non-finite value by ignoring it
            // when working out its range.
            "/" => left / right,
            "min" => left.min(right),
            "max" => left.max(right),
            _ => *left,
        };
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let mut products = Products::new();
    products.insert(
        "result",
        DataArray::numeric(Dtype::Float32, vec![pair.a.len() as u64], bytes).into(),
    );
    products.into()
}

fn run_logic(request: &Request) -> Outcome {
    let op = text(&request.params, "op", "and").to_string();
    // No literal: a boolean connective against a constant is either the input
    // or a constant, and neither is worth a control that looks like a choice.
    let pair = match read(request, None) {
        Ok(pair) => pair,
        Err(why) => return Outcome::refused(why),
    };

    // Non-zero is true, matching how a mask has always been read.
    let a: Vec<bool> = pair.a.iter().map(|value| *value != 0.0).collect();

    if op == "not" {
        return mask(a.iter().map(|value| u8::from(!value)).collect()).into();
    }
    let Some(b) = pair.b else {
        return Outcome::refused(format!(
            "\"{op}\" needs two masks, and nothing is bound to its second input"
        ));
    };

    let values = a
        .iter()
        .enumerate()
        .map(|(index, left)| {
            let right = at(&b, index) != 0.0;
            let kept = match op.as_str() {
                "and" => *left && right,
                "or" => *left || right,
                "xor" => *left != right,
                _ => *left,
            };
            u8::from(kept)
        })
        .collect();
    mask(values).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::registry::{ParamMap, ParamValue};
    use bevy::platform::collections::HashMap;

    fn array(values: &[f32]) -> DataArray {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DataArray::numeric(Dtype::Float32, vec![values.len() as u64], bytes)
    }

    fn request(inputs: &[(&'static str, &[f32])], params: &[(&str, ParamValue)]) -> Request {
        let mut map = ParamMap::new();
        for (key, value) in params {
            map.insert((*key).to_string(), value.clone());
        }
        let mut bound = HashMap::new();
        for (id, values) in inputs {
            bound.insert(*id, array(values));
        }
        Request {
            params: map,
            inputs: bound,
        }
    }

    fn bytes_of(outcome: &Outcome, id: &str) -> Vec<u8> {
        outcome.products[id].array().expect("an array").data.clone()
    }

    #[test]
    fn a_comparison_against_a_literal_makes_a_mask() {
        let out = run_compare(&request(
            &[("a", &[1.0, 5.0, 9.0])],
            &[
                ("op", ParamValue::Text(">".into())),
                ("value", ParamValue::Float(4.0)),
            ],
        ));
        assert_eq!(bytes_of(&out, "mask"), vec![0, 1, 1]);
    }

    #[test]
    fn a_comparison_against_another_array_is_elementwise() {
        let out = run_compare(&request(
            &[("a", &[1.0, 5.0]), ("b", &[2.0, 2.0])],
            &[("op", ParamValue::Text("<".into()))],
        ));
        assert_eq!(bytes_of(&out, "mask"), vec![1, 0]);
    }

    /// The routine mistake, and the one that has to say what it found.
    #[test]
    fn mismatched_lengths_are_refused_naming_both() {
        let out = run_compare(&request(
            &[("a", &[1.0, 2.0, 3.0]), ("b", &[1.0, 2.0])],
            &[],
        ));
        let problem = out.problem.expect("should have refused");
        assert!(problem.contains('3') && problem.contains('2'), "{problem}");
    }

    /// One value on the right applies to everything on the left, which is what
    /// makes "every atom in chain 1" one node rather than a broadcast step.
    #[test]
    fn a_single_value_broadcasts() {
        let out = run_compare(&request(
            &[("a", &[1.0, 2.0, 1.0]), ("b", &[1.0])],
            &[("op", ParamValue::Text("==".into()))],
        ));
        assert_eq!(bytes_of(&out, "mask"), vec![1, 0, 1]);
    }

    #[test]
    fn logic_combines_two_masks() {
        let out = run_logic(&request(
            &[("a", &[1.0, 1.0, 0.0]), ("b", &[1.0, 0.0, 0.0])],
            &[("op", ParamValue::Text("and".into()))],
        ));
        assert_eq!(bytes_of(&out, "mask"), vec![1, 0, 0]);
    }

    #[test]
    fn not_needs_only_one_mask() {
        let out = run_logic(&request(
            &[("a", &[1.0, 0.0])],
            &[("op", ParamValue::Text("not".into()))],
        ));
        assert_eq!(bytes_of(&out, "mask"), vec![0, 1]);
    }

    /// Every other connective does need two, and says so rather than treating
    /// the missing side as false — which would silently answer "nothing".
    #[test]
    fn and_without_a_second_mask_is_refused() {
        let out = run_logic(&request(
            &[("a", &[1.0, 0.0])],
            &[("op", ParamValue::Text("and".into()))],
        ));
        assert!(out.problem.is_some());
    }

    #[test]
    fn arithmetic_applies_the_literal_to_every_element() {
        let out = run_arithmetic(&request(
            &[("a", &[1.0, 2.0])],
            &[
                ("op", ParamValue::Text("*".into())),
                ("value", ParamValue::Float(3.0)),
            ],
        ));
        let values = DataArray::numeric(Dtype::Float32, vec![2], bytes_of(&out, "result")).to_f32();
        assert_eq!(values, vec![3.0, 6.0]);
    }

    /// A predicate over a vector field has several defensible readings, so it
    /// gets none of them.
    #[test]
    fn a_multi_component_array_is_refused() {
        let mut inputs = HashMap::new();
        inputs.insert(
            "a",
            DataArray::numeric(
                Dtype::Float32,
                vec![2, 3],
                [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
        );
        let out = run_compare(&Request {
            params: ParamMap::new(),
            inputs,
        });
        assert!(out.problem.expect("refused").contains("scalars"));
    }
}
