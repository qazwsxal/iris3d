//! Declared parameters, and the values that satisfy a declaration.
//!
//! A kind — an actor's or a filter's — states what it can be tuned by, and
//! that one declaration drives the interface's controls, the wire format and
//! the defaults. Reading a parameter goes through the accessors here rather
//! than indexing the map, so a client that sends nonsense gets a sensible
//! result instead of a panic.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use iris3d_data::array::{Dtype, Held};

/// A single tunable value on an actor.
///
/// Not `Copy`, because `Text` owns a `String`. Everything that reads a
/// parameter borrows it.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
    /// A name: a field to read, or one option out of a fixed set.
    Text(String),
    /// A fixed-length vector of numbers: an origin, a spacing, a size in
    /// samples.
    ///
    /// One variant covering every length rather than `Vec2`, `Vec3`, `Vec4` in
    /// turn. The length a parameter accepts is in its
    /// [`ParamKind::Vector`] declaration, so 2D work needs no new variant here —
    /// only a spec that asks for two components.
    ///
    /// Held as `f64` because it also carries counts: a grid's `dims` are
    /// integers, and `f32` stops representing those exactly at 16.8 million,
    /// which a 256³ grid is not far off in total samples.
    Vector(Vec<f64>),
    /// An uploaded array, by the handle [`DataStore`](super::DataStore) knows it
    /// by.
    ///
    /// Geometry is a parameter like any other, deliberately. An actor's arrays
    /// and its settings are edited by the same call, merged by the same rule,
    /// and generate their controls from the same declaration — rebinding the
    /// positions an actor draws is the same kind of operation as moving a
    /// slider, so it would be strange for it to be a different mechanism.
    Data(u64),
    /// Take the binding off an input, leaving it as though nothing was ever
    /// bound.
    ///
    /// A value meaning "no value", which reads oddly but is what the merge rule
    /// forces. Both `SetActor` and `SetFilter` take a **partial** map and leave
    /// anything absent alone, so a key's absence already means "unchanged" and
    /// cannot also mean "clear this". Without a way to say it, an optional input
    /// — `normals`, `colour`, `colour_field`, `vertices` — could be bound once
    /// and never let go, from any client, which was simply a missing operation
    /// rather than a decision.
    ///
    /// Only valid on an input, and refused on a required one: clearing what a
    /// kind cannot draw without leaves something that cannot draw, and
    /// `check_bindings` says so in the same words it uses for a missing input.
    Unset,
}

impl ParamValue {
    pub fn as_vector(&self) -> Option<&[f64]> {
        match self {
            ParamValue::Vector(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_data(&self) -> Option<u64> {
        match self {
            ParamValue::Data(id) => Some(*id),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f32> {
        match self {
            ParamValue::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ParamValue::Text(value) => Some(value),
            _ => None,
        }
    }
}

/// An actor's parameters, keyed by [`ParamSpec::id`].
pub type ParamMap = HashMap<String, ParamValue>;

/// Reads a float parameter, falling back when it is missing or the wrong type.
///
/// Backends call this rather than indexing, so a client that sends nonsense
/// gets a sensible render instead of a panic.
pub fn float(params: &ParamMap, id: &str, fallback: f32) -> f32 {
    params
        .get(id)
        .and_then(|value| value.as_float())
        .unwrap_or(fallback)
}

/// Reads a boolean parameter. See [`float`].
pub fn flag(params: &ParamMap, id: &str, fallback: bool) -> bool {
    params
        .get(id)
        .and_then(|value| value.as_bool())
        .unwrap_or(fallback)
}

/// Reads a text parameter — a field name, or a chosen option. See [`float`].
pub fn text<'a>(params: &'a ParamMap, id: &str, fallback: &'a str) -> &'a str {
    params
        .get(id)
        .and_then(|value| value.as_text())
        .unwrap_or(fallback)
}

/// Reads a bound array's handle. `None` means nothing is bound to that slot,
/// which is the normal state of an optional one. See [`float`].
pub fn data(params: &ParamMap, id: &str) -> Option<u64> {
    params.get(id).and_then(|value| value.as_data())
}

/// Reads a vector parameter of any declared length, padding with zeros if what
/// is stored is the wrong length. For a control that has to render something
/// whatever the map holds. See [`float`].
pub fn vector(params: &ParamMap, id: &str, components: usize) -> Vec<f64> {
    let mut values = params
        .get(id)
        .and_then(|value| value.as_vector())
        .map(<[f64]>::to_vec)
        .unwrap_or_default();
    values.resize(components, 0.0);
    values
}

/// Reads a three-component vector parameter. See [`float`].
pub fn vec3(params: &ParamMap, id: &str, fallback: Vec3) -> Vec3 {
    params
        .get(id)
        .and_then(|value| value.as_vector())
        .filter(|values| values.len() == 3)
        .map(|values| Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32))
        .unwrap_or(fallback)
}

/// Reads a three-component vector parameter as counts. See [`float`].
///
/// Negative or fractional components round towards a usable count rather than
/// wrapping: a dimension of -1 is nonsense, and silently becoming 4294967295
/// would ask for an allocation the size of the address space.
pub fn uvec3(params: &ParamMap, id: &str, fallback: UVec3) -> UVec3 {
    params
        .get(id)
        .and_then(|value| value.as_vector())
        .filter(|values| values.len() == 3)
        .map(|values| {
            let axis = |value: f64| value.round().clamp(0.0, u32::MAX as f64) as u32;
            UVec3::new(axis(values[0]), axis(values[1]), axis(values[2]))
        })
        .unwrap_or(fallback)
}

/// What a parameter is, which decides both its control and its valid range.
#[derive(Debug, Clone, Copy)]
pub enum ParamKind {
    Float {
        default: f32,
        min: f32,
        max: f32,
        /// Slide on a log scale. Right for anything spanning orders of
        /// magnitude, like a point size that is useful from 0.001 to 1.
        logarithmic: bool,
    },
    Bool {
        default: bool,
    },
    /// Free text, where no fixed set of options exists.
    ///
    /// Distinct from [`Choice`](ParamKind::Choice), which is text out of a list
    /// the kind knows in advance. This is for a value only the data can supply —
    /// `match`'s residue names are the case that needed it, since the names in a
    /// structure are not knowable when the filter is declared.
    ///
    /// Deliberately *not* an expression: what a kind does with the string is its
    /// own business, and nothing here parses one. A kind that finds itself
    /// wanting a grammar wants a graph instead.
    Text {
        default: &'static str,
    },
    /// One option out of a fixed set, such as a rendering mode.
    Choice {
        options: &'static [&'static str],
        default: &'static str,
    },
    /// A fixed-length vector of numbers: an origin, a spacing, a size in
    /// samples.
    ///
    /// `components` is the length, so a 2D parameter is this same variant asking
    /// for two. Every component shares one range, which is true of everything
    /// wanted so far — a spacing is positive on all three axes, an origin
    /// unbounded on all three.
    Vector {
        components: usize,
        default: &'static [f64],
        min: f64,
        max: f64,
        /// Whole numbers only, for counts like a grid's dimensions. Clients
        /// should offer an integer control rather than a fractional one.
        integral: bool,
    },
    /// An array the kind reads: positions, indices, a scalar field.
    ///
    /// This is how a kind says what it needs in order to draw anything, and it
    /// replaces guessing from buffer names. A client asks `ListActorKinds`, sees
    /// that `points` wants `float32 [n, 3]` under `positions`, and binds an
    /// array it uploaded. Nothing infers a role from what an array was called.
    Array {
        /// Element types that will do. Empty accepts any.
        dtypes: &'static [Dtype],
        /// Shape, one entry per axis, where 0 accepts any length: positions is
        /// `[0, 3]`, a triangle index array `[0, 3]`, a scalar field `[0]`. An
        /// empty slice accepts any shape at all.
        shape: &'static [u64],
        /// Whether the kind can draw without it. Colour fields are optional;
        /// positions are not.
        required: bool,
        /// Whether new data here changes the *shape* of what is drawn.
        ///
        /// `true` for positions and connectivity: different numbers mean a
        /// different vertex count, so there is nothing to write in place and
        /// the whole thing is rebuilt. `false` for a per-element colour, which
        /// lands in the buffer that is already there.
        ///
        /// Declared rather than guessed from the input's name, for the same
        /// reason the shape is: a kind says what it needs, and nothing infers a
        /// role from what something is called. It is what lets dragging a
        /// colour-map slider repaint a protein instead of re-tessellating every
        /// atom and bond — see [`Dirty`](crate::draw::Dirty), which grades the
        /// two apart precisely because they differ by orders of magnitude.
        structural: bool,
    },
    /// A mesh the kind draws, as one handle: vertices, triangles, and whatever
    /// attributes came with them.
    ///
    /// Not an [`Array`](Self::Array) of a particular shape, because it is not
    /// numbers the kind decodes. It is geometry somebody else assembled, and
    /// every consumer **references** it rather than building its own — which is
    /// the whole reason it exists: a ribbon drawn as a lit surface and as an
    /// absorbing medium is one set of vertices, uploaded once.
    ///
    /// What it carries is described by
    /// [`GeometryMeta`](iris3d_data::array::GeometryMeta), and a kind reads it rather
    /// than declaring a requirement: `medium` wants normals only when its shell
    /// is on, so refusing a normal-less mesh at bind time would refuse the
    /// commoner case. A kind that cannot use what it was given says so when it
    /// draws.
    ///
    /// There is no `structural` here. Nothing downstream rebuilds when the
    /// vertices move: the consumer holds the same `Handle<Mesh>` it always did
    /// and Bevy re-uploads the asset underneath it.
    Geometry {
        /// Whether the kind can draw without it.
        required: bool,
    },
}

impl ParamKind {
    /// Whether this parameter binds data rather than carrying a setting.
    ///
    /// The two data kinds against the four settings. Asked wherever bindings are
    /// gathered, so that adding a third never means finding every `matches!`.
    pub fn is_input(self) -> bool {
        matches!(self, ParamKind::Array { .. } | ParamKind::Geometry { .. })
    }

    /// Whether the kind refuses to work without this input bound. `false` for
    /// anything that is not an input at all.
    pub fn is_required(self) -> bool {
        match self {
            ParamKind::Array { required, .. } | ParamKind::Geometry { required } => required,
            _ => false,
        }
    }

    /// The value to start from, or `None` for a parameter with nothing sensible
    /// to start from.
    ///
    /// Only the input kinds have none. There is no default array — handle 0 is a
    /// real array belonging to whoever uploaded first, so inventing one would
    /// silently draw somebody else's data.
    pub fn default_value(self) -> Option<ParamValue> {
        match self {
            ParamKind::Float { default, .. } => Some(ParamValue::Float(default)),
            ParamKind::Bool { default } => Some(ParamValue::Bool(default)),
            ParamKind::Choice { default, .. } => Some(ParamValue::Text(default.to_string())),
            ParamKind::Text { default } => Some(ParamValue::Text(default.to_string())),
            ParamKind::Vector { default, .. } => Some(ParamValue::Vector(default.to_vec())),
            ParamKind::Array { .. } | ParamKind::Geometry { .. } => None,
        }
    }

    /// Whether what the handle names may be bound here.
    ///
    /// Deliberately not part of [`sanitise`](Self::sanitise). Sanitising judges
    /// a value on its own and is called wherever a parameter is written;
    /// checking a binding needs the [`DataStore`](super::DataStore) to look up
    /// what the handle actually points at, and the store is not reachable from
    /// every one of those places. So the two checks stay separate: sanitise
    /// decides "is this the right *kind* of value", this decides "is that
    /// particular thing the right shape".
    ///
    /// An array bound where geometry belongs is refused here rather than at draw
    /// time, and says so plainly: the two are one handle space, so the mistake
    /// is easy to make and cheap to name.
    pub fn accepts(self, held: Held<'_>) -> Result<(), String> {
        let meta = match (self, held) {
            (ParamKind::Geometry { .. }, Held::Geometry(_)) => return Ok(()),
            (ParamKind::Geometry { .. }, Held::Array(_)) => {
                return Err("is an array but this input takes geometry".into());
            }
            (ParamKind::Array { .. }, Held::Geometry(_)) => {
                return Err("is geometry but this input takes an array".into());
            }
            (ParamKind::Array { .. }, Held::Array(meta)) => meta,
            _ => return Err("not an input parameter".into()),
        };
        let ParamKind::Array { dtypes, shape, .. } = self else {
            unreachable!("matched an array parameter above");
        };
        // An array with no elements has no element type to disagree about, so
        // its declared one is not evidence of anything.
        //
        // This exists for the outputs that cannot state a type in advance —
        // `gather` hands back whatever it was given, so what it will be is not
        // known until it has run once. Every filter output is allocated empty at
        // creation precisely so it can be bound before the first run; refusing
        // one here on a placeholder dtype would take that back for exactly the
        // filters that need it most.
        //
        // The dtype is still checked on anything that holds data, which is every
        // upload and every output of a filter that has run. What it does not do
        // is re-check *after* a run — a binding is validated once, and a run may
        // rewrite the array's dtype. That was already true before this.
        // Rank zero counts as empty too. An output that cannot state its shape
        // in advance declares `[]`, which is "not stated" rather than "no
        // axes" — the same thing `None` says about a dtype. Testing only for a
        // zero *axis* missed it, because `[]` has no axes to test.
        let empty = meta.shape.is_empty() || meta.shape.contains(&0);
        if !empty && !dtypes.is_empty() && !dtypes.contains(&meta.dtype) {
            return Err(format!(
                "is {} but this input takes {}",
                meta.dtype,
                dtypes
                    .iter()
                    .map(|dtype| dtype.to_string())
                    .collect::<Vec<_>>()
                    .join(" or ")
            ));
        }
        // `shape.is_empty()` here is the *input* declaring it takes any shape.
        // `empty` is the *held array* having nothing in it to judge — a filter
        // output before its first run — and it passes for the same reason the
        // dtype check above lets it through.
        if shape.is_empty() || empty {
            return Ok(());
        }
        let fits = shape.len() == meta.shape.len()
            && shape
                .iter()
                .zip(&meta.shape)
                .all(|(wanted, actual)| *wanted == 0 || wanted == actual);
        if !fits {
            return Err(format!(
                "has shape {:?} but this input takes {}",
                meta.shape,
                describe_shape(shape)
            ));
        }
        Ok(())
    }

    /// Clamps a value into the declared range, and rejects one of the wrong
    /// type outright so a bad client cannot install a `Bool` where a slider
    /// expects a number.
    pub fn sanitise(self, value: ParamValue) -> Option<ParamValue> {
        match (self, value) {
            (ParamKind::Float { min, max, .. }, ParamValue::Float(value)) => {
                Some(ParamValue::Float(value.clamp(min, max)))
            }
            (ParamKind::Bool { .. }, ParamValue::Bool(value)) => Some(ParamValue::Bool(value)),
            // An option outside the declared set is rejected rather than
            // clamped: there is no nearest valid choice to fall back to.
            (ParamKind::Choice { options, .. }, ParamValue::Text(value)) => options
                .contains(&value.as_str())
                .then_some(ParamValue::Text(value)),
            // Nothing to validate: the kind reading it decides what it means,
            // and an empty string is a legitimate "not set yet".
            (ParamKind::Text { .. }, ParamValue::Text(value)) => Some(ParamValue::Text(value)),
            // Wrong length is rejected rather than padded or truncated: a
            // two-component origin is a caller mistake, and guessing the third
            // would place the data somewhere nobody asked for.
            (
                ParamKind::Vector {
                    components,
                    min,
                    max,
                    integral,
                    ..
                },
                ParamValue::Vector(values),
            ) => (values.len() == components).then(|| {
                ParamValue::Vector(
                    values
                        .into_iter()
                        .map(|value| {
                            let value = value.clamp(min, max);
                            if integral { value.round() } else { value }
                        })
                        .collect(),
                )
            }),
            // Whether what the handle names *fits* is `accepts`, checked where
            // the store is reachable. Here it only has to be a handle rather
            // than a number somebody meant as a slider value.
            (ParamKind::Array { .. } | ParamKind::Geometry { .. }, ParamValue::Data(id)) => {
                Some(ParamValue::Data(id))
            }
            // Only an input can be cleared, and *whether* it may be is not
            // decided here: `check_bindings` refuses a required one along with
            // every other missing input, in the same words, from the one place
            // that knows the whole picture. Rejecting it here as well would
            // report the same mistake two different ways.
            (ParamKind::Array { .. } | ParamKind::Geometry { .. }, ParamValue::Unset) => {
                Some(ParamValue::Unset)
            }
            _ => None,
        }
    }
}

/// Renders a declared shape for an error message: `[n, 3]`, `[n]`, `[4, 4]`.
fn describe_shape(shape: &[u64]) -> String {
    let axes: Vec<String> = shape
        .iter()
        .map(|axis| {
            if *axis == 0 {
                "n".to_string()
            } else {
                axis.to_string()
            }
        })
        .collect();
    format!("[{}]", axes.join(", "))
}

/// One tunable parameter of an actor kind.
#[derive(Debug, Clone, Copy)]
pub struct ParamSpec {
    /// Stable identifier, used as the map key and on the wire.
    pub id: &'static str,
    /// What to call it in the interface.
    pub label: &'static str,
    pub kind: ParamKind,
}
