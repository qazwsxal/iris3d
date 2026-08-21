//! The scene: objects held in the Bevy world, and the commands that change them.
//!
//! An object is a place in the tree and a name, and nothing else. It holds no
//! data at all, which is what lets one array feed several actors and one actor
//! read arrays that arrived separately. Getting numbers into the scene and
//! putting a node in the tree are two operations, deliberately.
//!
//! Three parts:
//!
//! - [`registry`] — what an actor is: a kind id, its parameters, and the arrays
//!   it binds. An actor is an entity, not a module. Backends register the kinds,
//!   so this knows nothing about how anything is drawn.
//! - [`link`] — where an actor sits, which is not the same question as what it
//!   reads. An actor can be drawn in several places at once.
//! - [`pick`] — what a click in the 3D view resolved to. The event lives here
//!   because three unrelated layers read it and none should depend on the one
//!   that raises it.
//!
//! The arrays themselves are in [`crate::data`], flat and by handle: they belong
//! to no object, and holding them here would put "get these numbers in" and "put
//! a node in the tree" back together.
//!
//! Nothing here knows about gRPC, and nothing here draws. A rendering backend
//! plugs in by consuming actors and their bindings; which one is running is a
//! launch choice, and this crate is the same either way. Filters are above this
//! too — they read and write arrays and have no place in the tree, which is why
//! they own their own commands rather than riding on [`SceneCommand`].

use bevy::prelude::*;

pub mod actor_commands;
pub mod apply;
pub mod command;
pub mod data_commands;
pub mod link;
pub mod object_commands;
pub mod pick;
pub mod registry;
pub mod subset;

// Only what other modules reach for. The rest stays available under its own
// module path — this is a binary crate, so unused re-exports are just noise.
/// The bus carrying [`SceneCommand`]s. See [`Bus`](crate::bus::Bus).
pub type CommandBus = crate::bus::Bus<SceneCommand>;

pub use apply::apply_scene_commands;
pub use command::{Deleted, SceneCommand};
pub use link::{Parents, Placement};
pub use pick::Picked;

// Re-exported because a caller reaching for the scene almost always wants these
// in the same breath. One extra path to each, deliberately.
pub use crate::data::{BufferMeta, DataArray, DataStore, Dtype, HeldMeta, NamedBuffer};

/// Ceiling on how far the ancestor walk will climb before giving up. Guards
/// against a pre-existing malformed hierarchy sending validation into a loop.
pub(crate) const MAX_HIERARCHY_DEPTH: usize = 4096;

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        let bus = CommandBus::from_world(app.world());
        app.insert_resource(bus)
            // Registered here because the type lives here: `viewport::pick` writes
            // it, and the interface, `filter::source` and `grpc::watch` all read
            // it. A reader panics on an unregistered message type, so no one
            // reader can be the one to register it.
            .add_message::<Picked>()
            .init_asset::<DataArray>()
            .init_resource::<DataStore>()
            .add_systems(
                Update,
                // Order matters: the drain creates actors, style components and
                // bindings are derived from the parameters it wrote, and the
                // placements follow the parents it set.
                (
                    apply_scene_commands,
                    registry::apply_actor_params,
                    link::sync_placements,
                    link::apply_shown,
                )
                    .chain(),
            );
    }
}

/// A place in the scene tree, and a name for it.
///
/// It holds no data: an actor binds the arrays it draws, so what an object *is*
/// is only where it is.
///
/// Paired with a [`UniqueID`](crate::counter::UniqueID) carrying its handle, a transform, and whatever
/// actors and child objects hang under it.
#[derive(Component, Debug)]
pub struct SceneObject {
    pub name: String,
}

/// One held handle, described without its contents.
#[derive(Debug, Clone)]
pub struct DataSummary {
    pub id: u64,
    pub meta: HeldMeta,
}

/// A description of an object in the scene.
///
/// No buffers and no byte count: an object holds no data to describe. What is
/// resident is a question about the arrays, and `ListData` answers it.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub id: u64,
    pub name: String,
    /// Everything drawing here.
    pub actors: Vec<ActorSummary>,
    /// Parent in the scene tree, `None` for a root.
    pub parent: Option<u64>,
}

/// A description of one way something is being drawn.
#[derive(Debug, Clone)]
pub struct ActorSummary {
    pub id: u64,
    /// Registered kind id — see [`ActorRegistry`](registry::ActorRegistry).
    pub kind: String,
    /// Every object it is drawn under, in handle order.
    ///
    /// Any number, including none. One actor under several objects is one
    /// drawing appearing in several places — changed once, changed everywhere
    /// — and an empty list draws nothing, which is where deleting the last
    /// object it was under leaves it.
    pub parents: Vec<u64>,
    pub params: crate::model::ParamMap,
    pub visible: bool,
}

/// A way of drawing that a backend has registered, described for a client.
#[derive(Debug, Clone)]
pub struct KindSummary {
    pub id: String,
    pub label: String,
    pub params: &'static [crate::model::ParamSpec],
}

#[cfg(test)]
mod tests;
