//! Which ways of drawing exist, and what can be tuned about each.
//!
//! Representation kinds used to be variants of an enum here, which meant this
//! module had to know every way of drawing anything — and had to carry a list
//! of which ones a backend could actually honour, a fact only the backend knows.
//! Backends register their kinds instead, so a kind exists exactly when
//! something can draw it and a new backend needs no edit here.
//!
//! Each kind declares its parameters, and that one declaration is what the UI
//! builds controls from, what the wire format carries, and what fills in
//! defaults. A representation's parameters live in [`RepresentationParams`] as
//! the single source of truth; [`apply_representation_params`] regenerates the
//! backend's own typed component from them whenever they change. Backends
//! therefore read plain typed fields and never touch the map, and nothing has
//! to keep two copies in agreement — the derived one is rewritten, never
//! edited.

use bevy::ecs::system::EntityCommands;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::DatasetKind;

/// A single tunable value on a representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
}

impl ParamValue {
    pub fn as_float(self) -> Option<f32> {
        match self {
            ParamValue::Float(value) => Some(value),
            ParamValue::Bool(_) => None,
        }
    }

    pub fn as_bool(self) -> Option<bool> {
        match self {
            ParamValue::Bool(value) => Some(value),
            ParamValue::Float(_) => None,
        }
    }
}

/// A representation's parameters, keyed by [`ParamSpec::id`].
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
}

impl ParamKind {
    pub fn default_value(self) -> ParamValue {
        match self {
            ParamKind::Float { default, .. } => ParamValue::Float(default),
            ParamKind::Bool { default } => ParamValue::Bool(default),
        }
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
            _ => None,
        }
    }
}

/// One tunable parameter of a representation kind.
#[derive(Debug, Clone, Copy)]
pub struct ParamSpec {
    /// Stable identifier, used as the map key and on the wire.
    pub id: &'static str,
    /// What to call it in the interface.
    pub label: &'static str,
    pub kind: ParamKind,
}

/// A way of drawing something, as declared by the backend that draws it.
pub struct RepresentationKind {
    /// Stable identifier — `"points"`, `"ball-and-stick"`. Goes over the wire.
    pub id: &'static str,
    pub label: &'static str,
    /// Whether this kind can draw a dataset of the given shape.
    pub supports: fn(DatasetKind) -> bool,
    pub params: &'static [ParamSpec],
    /// Writes the backend's own typed style component from the parameters.
    ///
    /// Called for a complete map, so an implementation may read every declared
    /// parameter and expect it to be there.
    pub apply: fn(&mut EntityCommands, &ParamMap),
}

impl RepresentationKind {
    pub fn spec(&self, id: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|spec| spec.id == id)
    }

    pub fn defaults(&self) -> ParamMap {
        self.params
            .iter()
            .map(|spec| (spec.id.to_string(), spec.kind.default_value()))
            .collect()
    }

    /// A complete, in-range parameter map built from whatever was supplied.
    ///
    /// Missing parameters take their default, out-of-range ones are clamped,
    /// and anything not declared is dropped. Every route into the scene goes
    /// through here, so no backend has to defend against a partial or hostile
    /// map.
    // Unused until a client can supply a map at all; the UI edits one value at
    // a time and goes through `ParamKind::sanitise` directly.
    #[allow(dead_code)]
    pub fn normalise(&self, given: &ParamMap) -> ParamMap {
        self.params
            .iter()
            .map(|spec| {
                let value = given
                    .get(spec.id)
                    .and_then(|value| spec.kind.sanitise(*value))
                    .unwrap_or_else(|| spec.kind.default_value());
                (spec.id.to_string(), value)
            })
            .collect()
    }
}

/// Every representation kind some backend has registered.
///
/// Order is registration order, and it decides which kind an upload is drawn
/// with: [`default_for`](Self::default_for) takes the first that supports the
/// dataset. Backends registering in `DrawPlugin` therefore also declare a
/// preference.
#[derive(Resource, Default)]
pub struct RepresentationRegistry(Vec<RepresentationKind>);

impl RepresentationRegistry {
    /// Adds a kind. A duplicate id replaces the earlier registration, so a
    /// backend can deliberately take over a name from another.
    pub fn register(&mut self, kind: RepresentationKind) {
        if let Some(existing) = self.0.iter_mut().find(|existing| existing.id == kind.id) {
            warn!("draw: representation kind \"{}\" re-registered", kind.id);
            *existing = kind;
            return;
        }
        self.0.push(kind);
    }

    pub fn get(&self, id: &str) -> Option<&RepresentationKind> {
        self.0.iter().find(|kind| kind.id == id)
    }

    // Wanted by the RPC that lets a client ask what kinds exist rather than
    // carrying its own table of them.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &RepresentationKind> {
        self.0.iter()
    }

    /// The kinds that can draw this dataset, in registration order.
    pub fn for_dataset(&self, dataset: DatasetKind) -> impl Iterator<Item = &RepresentationKind> {
        self.0.iter().filter(move |kind| (kind.supports)(dataset))
    }

    /// How to draw an upload that did not ask for anything specific.
    pub fn default_for(&self, dataset: DatasetKind) -> Option<&RepresentationKind> {
        self.for_dataset(dataset).next()
    }
}

/// Which registered kind a representation is.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentationKindId(pub &'static str);

/// A representation's parameters — the authoritative copy.
///
/// Always complete and in range: everything that writes it goes through
/// [`RepresentationKind::normalise`] first.
#[derive(Component, Debug, Clone, Default, PartialEq)]
pub struct RepresentationParams(pub ParamMap);

/// Regenerates backends' typed style components from the parameters.
///
/// `Changed` covers insertion, so a representation gets its style component on
/// the tick it is spawned. Backends never write these components, which is what
/// keeps the map authoritative rather than merely one of two opinions.
pub fn apply_representation_params(
    mut commands: Commands,
    registry: Res<RepresentationRegistry>,
    changed: Query<
        (Entity, &RepresentationKindId, &RepresentationParams),
        Changed<RepresentationParams>,
    >,
) {
    for (entity, kind, params) in &changed {
        let Some(registered) = registry.get(kind.0) else {
            warn!("draw: no backend registered for representation kind \"{}\"", kind.0);
            continue;
        };
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

    fn kind() -> RepresentationKind {
        RepresentationKind {
            id: "test",
            label: "test",
            supports: |dataset| dataset == DatasetKind::Points,
            params: SPECS,
            apply: |_, _| {},
        }
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
    /// backend that would then have to cope with them.
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

    #[test]
    fn the_first_registered_supporting_kind_is_the_default() {
        let mut registry = RepresentationRegistry::default();
        registry.register(kind());
        registry.register(RepresentationKind {
            id: "second",
            ..kind()
        });

        assert_eq!(
            registry.default_for(DatasetKind::Points).map(|kind| kind.id),
            Some("test")
        );
        assert_eq!(registry.for_dataset(DatasetKind::Points).count(), 2);
        assert!(registry.default_for(DatasetKind::Mesh).is_none());
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
        app.init_resource::<RepresentationRegistry>();
        app.add_systems(Update, apply_representation_params);
        app.world_mut()
            .resource_mut::<RepresentationRegistry>()
            .register(RepresentationKind {
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
            .spawn((
                RepresentationKindId("test"),
                RepresentationParams(kind().defaults()),
            ))
            .id();

        app.update();
        assert_eq!(
            app.world().get::<TestStyle>(entity),
            Some(&TestStyle {
                size: 0.05,
                shaded: true
            }),
            "a representation should have its style on the tick it appears"
        );

        app.world_mut()
            .get_mut::<RepresentationParams>(entity)
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

    /// A kind no backend registered leaves the entity without a style, so the
    /// backends simply never match it. Nothing should panic on the way there.
    #[test]
    fn an_unregistered_kind_is_survivable() {
        let mut app = app();
        let entity = app
            .world_mut()
            .spawn((
                RepresentationKindId("nothing-draws-this"),
                RepresentationParams::default(),
            ))
            .id();

        app.update();
        assert!(app.world().get::<TestStyle>(entity).is_none());
    }

    #[test]
    fn re_registering_replaces_rather_than_duplicates() {
        let mut registry = RepresentationRegistry::default();
        registry.register(kind());
        registry.register(RepresentationKind {
            label: "replaced",
            ..kind()
        });

        assert_eq!(registry.iter().count(), 1);
        assert_eq!(registry.get("test").map(|kind| kind.label), Some("replaced"));
    }
}
