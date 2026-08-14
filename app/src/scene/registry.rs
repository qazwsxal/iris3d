//! Which ways of drawing exist, and what can be tuned about each.
//!
//! Actor kinds used to be variants of an enum here, which meant this module
//! had to know every way of drawing anything — and had to carry a list of
//! which ones a backend could actually honour, a fact only the backend knows.
//! Backends register their kinds instead, so a kind exists exactly when
//! something can draw it and a new backend needs no edit here.
//!
//! Each kind declares its parameters, and that one declaration is what the UI
//! builds controls from, what the wire format carries, and what fills in
//! defaults. An actor's parameters live in [`ActorParams`] as the single
//! source of truth; [`apply_actor_params`] regenerates the kind's own typed
//! component from them whenever they change. A kind therefore reads plain typed
//! fields and never touches the map, and nothing has to keep two copies in
//! agreement — the derived one is rewritten, never edited.

use bevy::ecs::system::EntityCommands;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::data::{BufferMeta, Dtype};

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
    },
}

impl ParamKind {
    /// The value to start from, or `None` for a parameter with nothing sensible
    /// to start from.
    ///
    /// Only [`Array`](Self::Array) has none. There is no default array — handle
    /// 0 is a real array belonging to whoever uploaded first, so inventing one
    /// would silently draw somebody else's data.
    pub fn default_value(self) -> Option<ParamValue> {
        match self {
            ParamKind::Float { default, .. } => Some(ParamValue::Float(default)),
            ParamKind::Bool { default } => Some(ParamValue::Bool(default)),
            ParamKind::Choice { default, .. } => Some(ParamValue::Text(default.to_string())),
            ParamKind::Vector { default, .. } => Some(ParamValue::Vector(default.to_vec())),
            ParamKind::Array { .. } => None,
        }
    }

    /// Whether an array of this description may be bound here.
    ///
    /// Deliberately not part of [`sanitise`](Self::sanitise). Sanitising judges
    /// a value on its own and is called wherever a parameter is written;
    /// checking a binding needs the [`DataStore`](super::DataStore) to look up
    /// what the handle actually points at, and the store is not reachable from
    /// every one of those places. So the two checks stay separate: sanitise
    /// decides "is this the right *kind* of value", this decides "is that
    /// particular array the right shape".
    pub fn accepts(self, meta: &BufferMeta) -> Result<(), String> {
        let ParamKind::Array { dtypes, shape, .. } = self else {
            return Err("not an array parameter".into());
        };
        if !dtypes.is_empty() && !dtypes.contains(&meta.dtype) {
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
        if shape.is_empty() {
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
            // Whether the array *fits* is `accepts`, checked where the store is
            // reachable. Here it only has to be a handle rather than a number
            // somebody meant as a slider value.
            (ParamKind::Array { .. }, ParamValue::Data(id)) => Some(ParamValue::Data(id)),
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

/// A way of drawing something, as declared by the backend that draws it.
pub struct ActorKind {
    /// Stable identifier — `"points"`, `"ball-and-stick"`. Goes over the wire.
    pub id: &'static str,
    pub label: &'static str,

    pub params: &'static [ParamSpec],
    /// Writes the kind's own typed style component from the parameters.
    ///
    /// Called for a complete map, so an implementation may read every declared
    /// parameter and expect it to be there.
    pub apply: fn(&mut EntityCommands, &ParamMap),
}

impl ActorKind {
    pub fn spec(&self, id: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|spec| spec.id == id)
    }

    /// Every input this kind reads an array from, required or not.
    pub fn inputs(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params
            .iter()
            .filter(|spec| matches!(spec.kind, ParamKind::Array { .. }))
    }

    /// Settings at their starting values. Array inputs are absent: they have no
    /// default, so a new actor's map is complete only once they are bound.
    pub fn defaults(&self) -> ParamMap {
        self.params
            .iter()
            .filter_map(|spec| Some((spec.id.to_string(), spec.kind.default_value()?)))
            .collect()
    }

    /// A complete, in-range parameter map built from whatever was supplied.
    ///
    /// Missing parameters take their default, out-of-range ones are clamped,
    /// and anything not declared is dropped. Every route into the scene goes
    /// through here, so no kind has to defend against a partial or hostile
    /// map.
    // Unused until a client can supply a map at all; the UI edits one value at
    // a time and goes through `ParamKind::sanitise` directly.
    #[allow(dead_code)]
    pub fn normalise(&self, given: &ParamMap) -> ParamMap {
        self.params
            .iter()
            .filter_map(|spec| {
                // An unbound array stays unbound rather than becoming some
                // arbitrary handle, so the map says truthfully what is missing.
                let value = given
                    .get(spec.id)
                    .cloned()
                    .and_then(|value| spec.kind.sanitise(value))
                    .or_else(|| spec.kind.default_value())?;
                Some((spec.id.to_string(), value))
            })
            .collect()
    }
}

/// Every actor kind the running backend has registered.
///
/// Only one backend runs, so this holds one pathway's kinds and nothing has to
/// be filtered by which. Order is registration order, and it is presentation
/// order only — what the UI lists and what `ListActorKinds` returns. Nothing
/// here picks a kind on a caller's behalf: the registry answers what exists,
/// and the caller decides.
#[derive(Resource)]
pub struct ActorRegistry {
    kinds: Vec<ActorKind>,
    backend: &'static str,
}

impl Default for ActorRegistry {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            // Overwritten before any backend registers. Only stays this way in
            // tests that exercise the scene with no pathway added at all.
            backend: "no",
        }
    }
}

impl ActorRegistry {
    /// Adds a kind. A duplicate id replaces the earlier registration, so a
    /// backend can deliberately take over a name from another.
    pub fn register(&mut self, kind: ActorKind) {
        if let Some(existing) = self.kinds.iter_mut().find(|existing| existing.id == kind.id) {
            warn!("draw: actor kind \"{}\" re-registered", kind.id);
            *existing = kind;
            return;
        }
        self.kinds.push(kind);
    }

    /// Records which backend these kinds came from, for messages that should
    /// say which pathway refused. A name rather than the type, so `scene` needs
    /// no knowledge of the backends themselves.
    pub fn served_by(&mut self, backend: &'static str) {
        self.backend = backend;
    }

    /// The running backend's name.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn get(&self, id: &str) -> Option<&ActorKind> {
        self.kinds.iter().find(|kind| kind.id == id)
    }

    // Wanted by the RPC that lets a client ask what kinds exist rather than
    // carrying its own table of them.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &ActorKind> {
        self.kinds.iter()
    }
}

/// Which registered kind an actor is.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActorKindId(pub &'static str);

/// An actor's parameters — the authoritative copy.
///
/// Always complete and in range: everything that writes it goes through
/// [`ActorKind::normalise`] first.
#[derive(Component, Debug, Clone, Default, PartialEq)]
pub struct ActorParams(pub ParamMap);

/// The arrays an actor has bound, by input id.
///
/// Derived from [`ActorParams`], like a kind's style component, and for the
/// same reason: a kind reads a plain typed field rather than searching the
/// map. One component for every kind rather than one per kind, because a
/// binding means the same thing everywhere and every kind resolves it the
/// same way — look the handle up in [`DataStore`](super::DataStore).
///
/// Written only when it actually differs, which is what keeps a slider drag from
/// invalidating geometry: `apply_actor_params` runs on any parameter change, and
/// an unconditional insert would mark this `Changed` every time.
#[derive(Component, Debug, Clone, Default, PartialEq)]
pub struct Bindings(pub HashMap<&'static str, u64>);

impl Bindings {
    pub fn get(&self, input: &str) -> Option<u64> {
        self.0.get(input).copied()
    }
}

/// Regenerates the kinds' typed style components from the parameters.
///
/// `Changed` covers insertion, so an actor gets its style component on the
/// tick it is spawned. A kind never writes these components, which is what
/// keeps the map authoritative rather than merely one of two opinions.
pub fn apply_actor_params(
    mut commands: Commands,
    registry: Res<ActorRegistry>,
    changed: Query<(Entity, &ActorKindId, &ActorParams, Option<&Bindings>), Changed<ActorParams>>,
) {
    for (entity, kind, params, bound) in &changed {
        let Some(registered) = registry.get(kind.0) else {
            warn!(
                "draw: the {} backend registered no actor kind \"{}\"",
                registry.backend(),
                kind.0
            );
            continue;
        };

        // Bindings before the style component, so a kind that reads both in
        // the same tick sees them agree.
        let wanted = Bindings(
            registered
                .inputs()
                .filter_map(|spec| Some((spec.id, data(&params.0, spec.id)?)))
                .collect(),
        );
        // Only when it differs: this system runs on any parameter change, and
        // an unconditional insert would mark the bindings `Changed` on every
        // slider drag, throwing away the geometry to rebuild it identically.
        if bound != Some(&wanted) {
            commands.entity(entity).insert(wanted);
        }

        let mut entity = commands.entity(entity);
        (registered.apply)(&mut entity, &params.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[ParamSpec] = &[
        ParamSpec {
            id: "size",
            label: "size",
            kind: ParamKind::Float {
                default: 0.05,
                min: 0.001,
                max: 1.0,
                logarithmic: true,
            },
        },
        ParamSpec {
            id: "shaded",
            label: "shaded",
            kind: ParamKind::Bool { default: true },
        },
    ];

    fn kind() -> ActorKind {
        ActorKind {
            id: "test",
            label: "test",
            params: SPECS,
            apply: |_, _| {},
        }
    }

    const WITH_INPUTS: &[ParamSpec] = &[
        ParamSpec {
            id: "positions",
            label: "positions",
            kind: ParamKind::Array {
                dtypes: &[Dtype::Float32],
                shape: &[0, 3],
                required: true,
            },
        },
        ParamSpec {
            id: "scalars",
            label: "scalars",
            kind: ParamKind::Array {
                dtypes: &[],
                shape: &[0],
                required: false,
            },
        },
    ];

    fn with_inputs() -> ActorKind {
        ActorKind {
            id: "bound",
            label: "bound",
            params: WITH_INPUTS,
            apply: |_, _| {},
        }
    }

    fn meta(dtype: Dtype, shape: &[u64]) -> BufferMeta {
        BufferMeta {
            name: "whatever".into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    const VECTORS: &[ParamSpec] = &[
        ParamSpec {
            id: "origin",
            label: "origin",
            kind: ParamKind::Vector {
                components: 3,
                default: &[0.0, 0.0, 0.0],
                min: -1e6,
                max: 1e6,
                integral: false,
            },
        },
        ParamSpec {
            id: "dims",
            label: "dimensions",
            kind: ParamKind::Vector {
                components: 3,
                default: &[1.0, 1.0, 1.0],
                min: 1.0,
                max: 4096.0,
                integral: true,
            },
        },
    ];

    /// A vector of the wrong length is refused outright. Padding a
    /// two-component origin would place the data on a third axis nobody named,
    /// and truncating a four-component one would silently drop what was sent.
    #[test]
    fn a_vector_of_the_wrong_length_is_refused() {
        let origin = VECTORS[0].kind;
        assert_eq!(
            origin.sanitise(ParamValue::Vector(vec![1.0, 2.0, 3.0])),
            Some(ParamValue::Vector(vec![1.0, 2.0, 3.0]))
        );
        assert_eq!(origin.sanitise(ParamValue::Vector(vec![1.0, 2.0])), None);
        assert_eq!(
            origin.sanitise(ParamValue::Vector(vec![1.0, 2.0, 3.0, 4.0])),
            None
        );
        // Still type-checked: a slider value is not a vector.
        assert_eq!(origin.sanitise(ParamValue::Float(1.0)), None);
    }

    /// Components clamp to the declared range, and an integral vector rounds —
    /// a count of 2.5 samples is not a thing to allocate.
    #[test]
    fn vector_components_clamp_and_counts_round() {
        assert_eq!(
            VECTORS[0]
                .kind
                .sanitise(ParamValue::Vector(vec![-1e9, 0.5, 1e9])),
            Some(ParamValue::Vector(vec![-1e6, 0.5, 1e6]))
        );
        assert_eq!(
            VECTORS[1]
                .kind
                .sanitise(ParamValue::Vector(vec![2.5, 0.0, 99999.0])),
            Some(ParamValue::Vector(vec![3.0, 1.0, 4096.0]))
        );
    }

    /// Reading a count back is total: a wrong-length or negative value falls
    /// back rather than wrapping to four billion.
    #[test]
    fn reading_counts_never_wraps() {
        let mut params = ParamMap::default();
        params.insert("dims".into(), ParamValue::Vector(vec![-3.0, 2.6, 4.0]));
        assert_eq!(uvec3(&params, "dims", UVec3::ONE), UVec3::new(0, 3, 4));

        params.insert("dims".into(), ParamValue::Vector(vec![2.0, 2.0]));
        assert_eq!(
            uvec3(&params, "dims", UVec3::splat(7)),
            UVec3::splat(7),
            "the wrong length is not a partial answer"
        );
    }

    /// An input accepts on element type and shape, and 0 in a declared shape
    /// means any length. The name is never consulted: an array called anything
    /// at all binds, which is the whole point of binding over inference.
    #[test]
    fn an_input_accepts_the_shape_it_declared() {
        let positions = WITH_INPUTS[0].kind;
        assert!(positions.accepts(&meta(Dtype::Float32, &[500, 3])).is_ok());
        assert!(positions.accepts(&meta(Dtype::Float32, &[1, 3])).is_ok());

        // Wrong element type, wrong component count, wrong rank.
        assert!(positions.accepts(&meta(Dtype::Float64, &[500, 3])).is_err());
        assert!(positions.accepts(&meta(Dtype::Float32, &[500, 2])).is_err());
        assert!(positions.accepts(&meta(Dtype::Float32, &[500])).is_err());

        // An empty dtype list takes anything numeric.
        let scalars = WITH_INPUTS[1].kind;
        assert!(scalars.accepts(&meta(Dtype::Uint8, &[8])).is_ok());
        assert!(scalars.accepts(&meta(Dtype::Float64, &[8])).is_ok());
        assert!(scalars.accepts(&meta(Dtype::Float64, &[8, 3])).is_err());
    }

    /// The error says what the input wanted, in its own terms, so a client can
    /// fix the call without reading the server.
    #[test]
    fn a_rejected_binding_explains_itself() {
        let positions = WITH_INPUTS[0].kind;
        let reason = positions
            .accepts(&meta(Dtype::Float32, &[500, 4]))
            .expect_err("four components is not three");
        assert!(reason.contains("[500, 4]"), "{reason}");
        assert!(reason.contains("[n, 3]"), "{reason}");
    }

    /// Settings have defaults; arrays do not. Handle 0 is a real array
    /// belonging to whoever uploaded first, so there is nothing safe to invent —
    /// an unbound input has to stay absent and be reported as missing.
    #[test]
    fn an_unbound_input_stays_absent() {
        let normalised = with_inputs().normalise(&ParamMap::default());
        assert!(normalised.is_empty(), "{normalised:?}");

        let mut given = ParamMap::default();
        given.insert("positions".into(), ParamValue::Data(7));
        // A slider value where an array belongs is refused, not coerced.
        given.insert("scalars".into(), ParamValue::Float(1.0));
        let normalised = with_inputs().normalise(&given);
        assert_eq!(normalised.get("positions"), Some(&ParamValue::Data(7)));
        assert_eq!(normalised.get("scalars"), None);
    }

    #[test]
    fn defaults_fill_in_what_was_not_given() {
        let normalised = kind().normalise(&ParamMap::default());
        assert_eq!(normalised.get("size"), Some(&ParamValue::Float(0.05)));
        assert_eq!(normalised.get("shaded"), Some(&ParamValue::Bool(true)));
    }

    /// A client is not to be trusted with the range; a point size of ten
    /// million is a hang, not a render.
    #[test]
    fn out_of_range_values_are_clamped() {
        let mut given = ParamMap::default();
        given.insert("size".into(), ParamValue::Float(1e7));
        assert_eq!(
            kind().normalise(&given).get("size"),
            Some(&ParamValue::Float(1.0))
        );
    }

    /// Wrong type and unknown name both fall back rather than reaching a
    /// kind that would then have to cope with them.
    #[test]
    fn nonsense_is_dropped() {
        let mut given = ParamMap::default();
        given.insert("size".into(), ParamValue::Bool(true));
        given.insert("nonexistent".into(), ParamValue::Float(1.0));

        let normalised = kind().normalise(&given);
        assert_eq!(normalised.get("size"), Some(&ParamValue::Float(0.05)));
        assert!(!normalised.contains_key("nonexistent"));
        assert_eq!(normalised.len(), 2);
    }

    /// Kinds come back in registration order, and every one of them is offered
    /// for every object. Which kinds *could* draw an object used to be filtered
    /// by a `supports(DatasetKind)` predicate; an actor's data is bound to it
    /// now, so the object it hangs under says nothing about what can draw it.
    #[test]
    fn kinds_come_back_in_registration_order() {
        let mut registry = ActorRegistry::default();
        registry.register(kind());
        registry.register(ActorKind {
            id: "second",
            ..kind()
        });

        let listed: Vec<&str> = registry.iter().map(|kind| kind.id).collect();
        assert_eq!(listed, ["test", "second"]);
    }

    /// Style component derived from the map, which is what lets the map be the
    /// only thing anything writes.
    #[derive(Component, Debug, PartialEq)]
    struct TestStyle {
        size: f32,
        shaded: bool,
    }

    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<ActorRegistry>();
        app.add_systems(Update, apply_actor_params);
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .register(ActorKind {
                apply: |entity, params| {
                    entity.insert(TestStyle {
                        size: float(params, "size", 0.05),
                        shaded: flag(params, "shaded", true),
                    });
                },
                ..kind()
            });
        app
    }

    #[test]
    fn parameters_become_a_style_component() {
        let mut app = app();
        let entity = app
            .world_mut()
            .spawn((ActorKindId("test"), ActorParams(kind().defaults())))
            .id();

        app.update();
        assert_eq!(
            app.world().get::<TestStyle>(entity),
            Some(&TestStyle {
                size: 0.05,
                shaded: true
            }),
            "an actor should have its style on the tick it appears"
        );

        app.world_mut()
            .get_mut::<ActorParams>(entity)
            .unwrap()
            .0
            .insert("size".into(), ParamValue::Float(0.5));
        app.update();
        assert_eq!(
            app.world().get::<TestStyle>(entity).map(|style| style.size),
            Some(0.5),
            "editing the map should rewrite the derived component"
        );
    }

    /// A kind nothing registered leaves the entity without a style, so the
    /// backend's systems simply never match it. Nothing should panic on the way
    /// there.
    #[test]
    fn an_unregistered_kind_is_survivable() {
        let mut app = app();
        let entity = app
            .world_mut()
            .spawn((ActorKindId("nothing-draws-this"), ActorParams::default()))
            .id();

        app.update();
        assert!(app.world().get::<TestStyle>(entity).is_none());
    }

    /// Which pathway registered these kinds, so a refusal can say so. Untouched
    /// it reads "no", which is only ever true in a test with no backend added.
    #[test]
    fn the_registry_names_the_backend_that_filled_it() {
        let mut registry = ActorRegistry::default();
        assert_eq!(registry.backend(), "no");
        registry.served_by("default");
        assert_eq!(registry.backend(), "default");
    }

    #[test]
    fn re_registering_replaces_rather_than_duplicates() {
        let mut registry = ActorRegistry::default();
        registry.register(kind());
        registry.register(ActorKind {
            label: "replaced",
            ..kind()
        });

        assert_eq!(registry.iter().count(), 1);
        assert_eq!(
            registry.get("test").map(|kind| kind.label),
            Some("replaced")
        );
    }
}
