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
//! source of truth; [`apply_actor_params`] regenerates the backend's own typed
//! component from them whenever they change. Backends therefore read plain
//! typed fields and never touch the map, and nothing has to keep two copies in
//! agreement — the derived one is rewritten, never edited.

use bevy::ecs::system::EntityCommands;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::DatasetKind;

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
}

impl ParamValue {
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
    /// Names a field on the source object.
    ///
    /// The empty string means "choose one", and what that resolves to is the
    /// backend's business — the same posture `ColorBy::field` takes. Nothing
    /// here checks that the name exists, because the object it must exist on is
    /// not known until draw time, and a field can appear or vanish after the
    /// parameter is set.
    Field,
    /// One option out of a fixed set, such as a rendering mode.
    Choice {
        options: &'static [&'static str],
        default: &'static str,
    },
}

impl ParamKind {
    pub fn default_value(self) -> ParamValue {
        match self {
            ParamKind::Float { default, .. } => ParamValue::Float(default),
            ParamKind::Bool { default } => ParamValue::Bool(default),
            ParamKind::Field => ParamValue::Text(String::new()),
            ParamKind::Choice { default, .. } => ParamValue::Text(default.to_string()),
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
            (ParamKind::Field, ParamValue::Text(value)) => Some(ParamValue::Text(value)),
            // An option outside the declared set is rejected rather than
            // clamped: there is no nearest valid choice to fall back to.
            (ParamKind::Choice { options, .. }, ParamValue::Text(value)) => options
                .contains(&value.as_str())
                .then_some(ParamValue::Text(value)),
            _ => None,
        }
    }
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
    /// Whether this kind can draw a dataset of the given shape.
    pub supports: fn(DatasetKind) -> bool,
    pub params: &'static [ParamSpec],
    /// Writes the backend's own typed style component from the parameters.
    ///
    /// Called for a complete map, so an implementation may read every declared
    /// parameter and expect it to be there.
    pub apply: fn(&mut EntityCommands, &ParamMap),
}

impl ActorKind {
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
                    .cloned()
                    .and_then(|value| spec.kind.sanitise(value))
                    .unwrap_or_else(|| spec.kind.default_value());
                (spec.id.to_string(), value)
            })
            .collect()
    }
}

/// Every actor kind some backend has registered.
///
/// Order is registration order, and it is presentation order only — what the
/// UI lists and what `ListActorKinds` returns. Nothing here picks a kind on a
/// caller's behalf: the registry answers what exists, and the caller decides.
#[derive(Resource, Default)]
pub struct ActorRegistry(Vec<ActorKind>);

impl ActorRegistry {
    /// Adds a kind. A duplicate id replaces the earlier registration, so a
    /// backend can deliberately take over a name from another.
    pub fn register(&mut self, kind: ActorKind) {
        if let Some(existing) = self.0.iter_mut().find(|existing| existing.id == kind.id) {
            warn!("draw: actor kind \"{}\" re-registered", kind.id);
            *existing = kind;
            return;
        }
        self.0.push(kind);
    }

    pub fn get(&self, id: &str) -> Option<&ActorKind> {
        self.0.iter().find(|kind| kind.id == id)
    }

    // Wanted by the RPC that lets a client ask what kinds exist rather than
    // carrying its own table of them.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = &ActorKind> {
        self.0.iter()
    }

    /// The kinds that can draw this dataset, in registration order.
    pub fn for_dataset(&self, dataset: DatasetKind) -> impl Iterator<Item = &ActorKind> {
        self.0.iter().filter(move |kind| (kind.supports)(dataset))
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

/// Regenerates backends' typed style components from the parameters.
///
/// `Changed` covers insertion, so an actor gets its style component on the
/// tick it is spawned. Backends never write these components, which is what
/// keeps the map authoritative rather than merely one of two opinions.
pub fn apply_actor_params(
    mut commands: Commands,
    registry: Res<ActorRegistry>,
    changed: Query<(Entity, &ActorKindId, &ActorParams), Changed<ActorParams>>,
) {
    for (entity, kind, params) in &changed {
        let Some(registered) = registry.get(kind.0) else {
            warn!("draw: no backend registered for actor kind \"{}\"", kind.0);
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

    fn kind() -> ActorKind {
        ActorKind {
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

    /// The registry reports what could draw a dataset, in the order it was
    /// registered, and stops there. It used to hand out the first of them as a
    /// default, which made registration order into a rendering decision.
    #[test]
    fn supporting_kinds_come_back_in_registration_order() {
        let mut registry = ActorRegistry::default();
        registry.register(kind());
        registry.register(ActorKind {
            id: "second",
            ..kind()
        });

        let supporting: Vec<&str> = registry
            .for_dataset(DatasetKind::Points)
            .map(|kind| kind.id)
            .collect();
        assert_eq!(supporting, ["test", "second"]);
        assert_eq!(registry.for_dataset(DatasetKind::Mesh).count(), 0);
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

    /// A kind no backend registered leaves the entity without a style, so the
    /// backends simply never match it. Nothing should panic on the way there.
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
