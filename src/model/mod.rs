//! What a client and the server agree about.
//!
//! Three things, and they are together because they are the shared vocabulary
//! of a request rather than any one layer's business:
//!
//! - [`param`] — what a kind declares it can be tuned by, and the values that
//!   satisfy a declaration. One declaration is what the interface builds
//!   controls from, what the wire format carries, and what fills in defaults.
//! - [`Bindings`] — which array an input is pointed at, derived from the
//!   parameters. An actor and a filter each carry one, and both resolve it the
//!   same way: look the handle up in the store.
//! - [`SceneError`] — the ways a request can be refused. Both the scene and the
//!   filter graph raise these, and the wire maps them to status codes, so the
//!   set has to sit below all three.
//!
//! Nothing here knows what a scene is or what a filter is. It knows what a
//! *declaration* is, and what it means for a value to satisfy one.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};

pub mod param;

pub use param::{
    ParamKind, ParamMap, ParamSpec, ParamValue, data, flag, float, normalise, text, uvec3, vec3,
    vector,
};

/// The arrays an actor has bound, by input id.
///
/// Derived from `ActorParams`, like a kind's style component, and for the
/// same reason: a kind reads a plain typed field rather than searching the
/// map. One component for every kind rather than one per kind, because a
/// binding means the same thing everywhere and every kind resolves it the
/// same way — look the handle up in [`DataStore`](crate::data::DataStore).
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    NoSuchObject(u64),
    NoSuchActor(u64),
    /// A handle that names no held array. Distinct from `NoSuchObject` because
    /// the three handle spaces share one sequence — passing an object handle
    /// where an array was wanted is a plausible mistake worth naming precisely.
    NoSuchData(u64),
    /// An input the kind cannot draw without was left unbound.
    MissingInput {
        kind: String,
        input: &'static str,
    },
    /// The bound array is the wrong element type or shape for the input.
    BadBinding {
        kind: String,
        input: &'static str,
        /// What is wrong, in the words of the input's own declaration.
        reason: String,
    },
    /// The running backend registered no kind by that name, so nothing could
    /// draw it.
    ///
    /// Carries the pathway as well as the name, because a kind can be perfectly
    /// real under another backend and simply absent here. Saying which one is
    /// running is the difference between "you mistyped it" and "you are on the
    /// wrong pathway".
    UnknownKind {
        kind: String,
        backend: &'static str,
    },
    NoSuchFilter(u64),
    /// No filter kind by that name. No backend to name, unlike
    /// [`UnknownKind`](Self::UnknownKind): a filter derives data and has no
    /// pipeline, so the same set exists in every build.
    UnknownFilterKind {
        kind: String,
    },
    /// Binding this would put a filter's own output somewhere in its own
    /// inputs.
    ///
    /// Refused rather than tolerated. A cycle is not a slow render: each run
    /// rewrites an array that marks the next one stale, so the graph never comes
    /// to rest and the app never sleeps.
    FilterCycle {
        filter: u64,
        input: &'static str,
    },
    /// A handle a filter is writing cannot be forgotten on its own.
    ///
    /// Releasing it would leave the filter producing into nothing, which looks
    /// exactly like a filter that has stopped working. Remove the filter.
    StillGenerated {
        data: u64,
        filter: u64,
    },
    /// The requested parent is the object itself or one of its descendants.
    ///
    /// Rejecting this is not optional: Bevy's transform propagation *panics* on
    /// a hierarchy cycle, so allowing one would let a client crash the
    /// application with two calls.
    WouldCycle {
        object: u64,
        parent: u64,
    },
}

impl Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::NoSuchObject(id) => write!(f, "no object with handle {id}"),
            SceneError::NoSuchActor(id) => {
                write!(f, "no actor with handle {id}")
            }
            SceneError::NoSuchData(id) => write!(f, "no uploaded array with handle {id}"),
            SceneError::MissingInput { kind, input } => write!(
                f,
                "actor kind \"{kind}\" cannot draw without an array bound to \"{input}\""
            ),
            SceneError::BadBinding {
                kind,
                input,
                reason,
            } => write!(
                f,
                "the array bound to \"{input}\" of actor kind \"{kind}\" {reason}"
            ),
            SceneError::UnknownKind { kind, backend } => write!(
                f,
                "the {backend} backend has no actor kind \"{kind}\" — ask \
                 ListActorKinds for the ones this pathway draws"
            ),
            SceneError::WouldCycle { object, parent } => write!(
                f,
                "object {object} cannot be parented to {parent}: {parent} is {object} \
                 or one of its descendants"
            ),
            SceneError::NoSuchFilter(id) => write!(f, "no filter with handle {id}"),
            SceneError::UnknownFilterKind { kind } => write!(
                f,
                "there is no filter kind \"{kind}\" — ask ListFilterKinds for the \
                 ones this build has"
            ),
            SceneError::FilterCycle { filter, input } => write!(
                f,
                "binding that to \"{input}\" would feed filter {filter} its own \
                 output, directly or through others"
            ),
            SceneError::StillGenerated { data, filter } => write!(
                f,
                "array {data} is an output of filter {filter} and cannot be released \
                 on its own — remove the filter instead"
            ),
        }
    }
}

impl std::error::Error for SceneError {}

/// Checks that every input a kind reads is bound, and bound to something it can
/// actually read.
///
/// Separate from [`ParamKind::sanitise`] on purpose. Sanitising judges a value
/// on its own and runs wherever a parameter is written; this needs the
/// [`DataStore`](crate::data::DataStore) to see what a handle points at, and the
/// store is not reachable from all of those places. So one answers "is this the
/// right sort of value" and the other "is that particular array or mesh the
/// right shape".
///
/// Arrays and meshes share a handle space, so this is also where binding one
/// where the other belongs is caught and named.
///
/// Takes the id and the inputs rather than a kind: an actor and a filter are
/// gated identically, and neither of their kind types is visible from here.
pub fn check_bindings<'a>(
    kind_id: &str,
    inputs: impl Iterator<Item = &'a ParamSpec>,
    params: &ParamMap,
    store: &crate::data::DataStore,
) -> Result<(), SceneError> {
    for spec in inputs {
        let required = spec.kind.is_required();
        match param::data(params, spec.id) {
            Some(id) => {
                let held = store.held(id).ok_or(SceneError::NoSuchData(id))?;
                spec.kind
                    .accepts(held)
                    .map_err(|reason| SceneError::BadBinding {
                        kind: kind_id.to_string(),
                        input: spec.id,
                        reason,
                    })?;
            }
            // An optional input left unbound is the normal case, not a fault.
            None if required => {
                return Err(SceneError::MissingInput {
                    kind: kind_id.to_string(),
                    input: spec.id,
                });
            }
            None => {}
        }
    }
    Ok(())
}
