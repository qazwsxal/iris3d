//! Which ways of drawing exist, and what can be tuned about each.
//!
//! Backends register their kinds, so a kind exists exactly when something can
//! draw it and a new backend needs no edit here. This module therefore knows
//! nothing about any particular way of drawing, and needs no list of which
//! kinds a backend can honour — a fact only the backend has.
//!
//! Each kind declares its parameters, and that one declaration is what the UI
//! builds controls from, what the wire format carries, and what fills in
//! defaults. An actor's parameters live in `ActorParams` as the single
//! source of truth; [`apply_actor_params`] regenerates the kind's own typed
//! component from them whenever they change. A kind therefore reads plain typed
//! fields and never touches the map, and nothing has to keep two copies in
//! agreement — the derived one is rewritten, never edited.

use bevy::ecs::system::EntityCommands;
use bevy::prelude::*;

use iris3d_model::{Bindings, ParamMap, ParamSpec, data};

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

    /// Every input this kind binds data to — array or geometry, required or not.
    pub fn inputs(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|spec| spec.kind.is_input())
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
    /// See [`iris3d_model::normalise`], which an actor kind and a filter kind
    /// share.
    pub fn normalise(&self, given: &ParamMap) -> ParamMap {
        iris3d_model::normalise(self.params, given)
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
        if let Some(existing) = self
            .kinds
            .iter_mut()
            .find(|existing| existing.id == kind.id)
        {
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

    use iris3d_data::array::{BufferMeta, Dtype, Held};
    use iris3d_model::{ParamKind, ParamValue, flag, float, uvec3};

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
                structural: true,
            },
        },
        ParamSpec {
            id: "scalars",
            label: "scalars",
            kind: ParamKind::Array {
                dtypes: &[],
                shape: &[0],
                required: false,
                structural: true,
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

    /// A mesh, for the checks that an input tells the two apart.
    fn geometry() -> iris3d_data::array::GeometryMeta {
        iris3d_data::array::GeometryMeta {
            name: "whatever".into(),
            vertices: 8,
            triangles: 4,
            normals: true,
            colours: false,
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
        let takes = |kind: ParamKind, dtype, shape: &[u64]| {
            kind.accepts(Held::Array(&meta(dtype, shape))).is_ok()
        };
        let positions = WITH_INPUTS[0].kind;
        assert!(takes(positions, Dtype::Float32, &[500, 3]));
        assert!(takes(positions, Dtype::Float32, &[1, 3]));

        // Wrong element type, wrong component count, wrong rank.
        assert!(!takes(positions, Dtype::Float64, &[500, 3]));
        assert!(!takes(positions, Dtype::Float32, &[500, 2]));
        assert!(!takes(positions, Dtype::Float32, &[500]));

        // An empty dtype list takes anything numeric.
        let scalars = WITH_INPUTS[1].kind;
        assert!(takes(scalars, Dtype::Uint8, &[8]));
        assert!(takes(scalars, Dtype::Float64, &[8]));
        assert!(!takes(scalars, Dtype::Float64, &[8, 3]));
    }

    /// The error says what the input wanted, in its own terms, so a client can
    /// fix the call without reading the server.
    #[test]
    fn a_rejected_binding_explains_itself() {
        let positions = WITH_INPUTS[0].kind;
        let reason = positions
            .accepts(Held::Array(&meta(Dtype::Float32, &[500, 4])))
            .expect_err("four components is not three");
        assert!(reason.contains("[500, 4]"), "{reason}");
        assert!(reason.contains("[n, 3]"), "{reason}");
    }

    /// Arrays and meshes share one handle space, so binding the wrong sort is an
    /// easy mistake — and one the caller is told about by name rather than
    /// finding out from a blank screen.
    #[test]
    fn an_input_refuses_the_other_sort_of_handle() {
        let positions = WITH_INPUTS[0].kind;
        let reason = positions
            .accepts(Held::Geometry(&geometry()))
            .expect_err("a mesh is not an array of positions");
        assert!(reason.contains("geometry"), "{reason}");

        let takes_geometry = ParamKind::Geometry { required: true };
        assert!(takes_geometry.accepts(Held::Geometry(&geometry())).is_ok());
        let reason = takes_geometry
            .accepts(Held::Array(&meta(Dtype::Float32, &[500, 3])))
            .expect_err("positions are not a mesh");
        assert!(reason.contains("array"), "{reason}");
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
    /// for every object. Nothing filters by what an object holds: an actor's
    /// data is bound to the actor, so the object it hangs under says nothing
    /// about what can draw it.
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
