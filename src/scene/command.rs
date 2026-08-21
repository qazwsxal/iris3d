//! What can be asked of the scene from outside the ECS.
//!
//! Each variant carries the channel its reply goes to, so the submitting task
//! can await an answer without the scene knowing who asked. Buffers are fully
//! assembled before they arrive, so the tick that applies a command never does
//! the transfer.
//!
//! Filters have commands of their own — a filter has no place
//! in the tree, and the two are applied by different systems.

use bevy::prelude::*;
use tokio::sync::oneshot;

use crate::data::NamedBuffer;
use crate::model::SceneError;

use super::{ActorSummary, DataSummary, KindSummary, ObjectSummary};

/// Work submitted to the scene from outside the ECS.
///
/// Each variant carries the channel its result should be written to, so the
/// submitting task can await a reply without the scene knowing who asked.
/// Buffers are fully assembled before they arrive here — the ECS tick that
/// applies a command never does the transfer.
#[derive(Debug)]
pub enum SceneCommand {
    /// Takes in arrays on their own. No object, no actor — the reply hands back
    /// a handle per array, and something else decides what they are for.
    UploadData {
        arrays: Vec<NamedBuffer>,
        reply: oneshot::Sender<Vec<DataSummary>>,
    },
    ListData {
        reply: oneshot::Sender<Vec<DataSummary>>,
    },
    /// Forgets arrays. Reports the handles that were held, so a caller learns
    /// which of them it was wrong about.
    ReleaseData {
        ids: Vec<u64>,
        reply: oneshot::Sender<Vec<u64>>,
    },
    /// Creates an object holding no data, for use as a grouping node.
    CreateObject {
        name: String,
        reply: oneshot::Sender<ObjectSummary>,
    },
    SetParent {
        id: u64,
        /// `None` detaches, making the object a root.
        parent: Option<u64>,
        keep_world_transform: bool,
        reply: oneshot::Sender<Result<(), SceneError>>,
    },
    /// Sets an object's local placement. Unset components are left alone.
    SetTransform {
        id: u64,
        translation: Option<Vec3>,
        rotation: Option<Quat>,
        scale: Option<Vec3>,
        reply: oneshot::Sender<Result<(), SceneError>>,
    },
    ListObjects {
        reply: oneshot::Sender<Vec<ObjectSummary>>,
    },
    DeleteObject {
        id: u64,
        /// Deletes exactly this node. Every child is detached and becomes a
        /// root, actors as much as objects.
        reply: oneshot::Sender<Deleted>,
    },

    /// Draws something under an object. Adds; never replaces.
    AddActor {
        /// The objects to draw under, whose transforms it inherits. Empty
        /// makes one, since an actor has no place of its own. *What* it draws
        /// is in `params`, as bindings.
        parents: Vec<u64>,
        /// Which registered kind draws it. Named by the caller, always: the
        /// server has no opinion on how a dataset should look.
        kind: String,
        /// Partial. Anything unset takes the kind's **default**, there being no
        /// previous value.
        params: crate::model::ParamMap,
        /// `None` draws the whole dataset.
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    SetActor {
        id: u64,
        /// Partial. Anything unset keeps its **current** value — the opposite
        /// of [`AddActor`](Self::AddActor), because here there is one.
        params: crate::model::ParamMap,
        visible: Option<bool>,
        /// Replace the set of objects it is drawn under. `None` leaves them
        /// alone; `Some(vec![])` takes it off screen without removing it.
        parents: Option<Vec<u64>>,
        reply: oneshot::Sender<Result<ActorSummary, SceneError>>,
    },
    RemoveActor {
        id: u64,
        reply: oneshot::Sender<bool>,
    },
    ListActors {
        /// Restrict to those drawing one object.
        parent: Option<u64>,
        reply: oneshot::Sender<Result<Vec<ActorSummary>, SceneError>>,
    },
    ListActorKinds {
        reply: oneshot::Sender<Vec<KindSummary>>,
    },
}

/// What a deletion took with it.
///
/// Only ever the object named. Actors under it are detached rather than
/// destroyed, so nothing else goes.
#[derive(Debug, Default, Clone)]
pub struct Deleted {
    pub objects: Vec<u64>,
}
