//! The app talking back: events out to whoever is listening.
//!
//! Every other RPC is a client asking a question. This is the one place the
//! server speaks first — somebody clicked something, and a script that wants to
//! know is told.
//!
//! # Which way the channel goes
//!
//! Commands travel ECS-wards over [`CommandBus`](crate::scene::bus::CommandBus), a crossbeam
//! channel drained each `Update`. Events travel the other way, and cannot use
//! the same mechanism: there is one command queue and many watchers, and a
//! watcher appears and disappears while the app is running.
//!
//! So this is a `tokio::sync::broadcast`. Every subscriber gets every event, a
//! late subscriber gets nothing that happened before it arrived, and — the part
//! that matters — **a subscriber that stops reading is dropped from, not queued
//! for**. A slow client must not be able to make the app hold frames of clicks
//! in memory, and `broadcast` gives that shape without a policy of its own: the
//! channel has a fixed depth and the laggard is told it lagged.
//!
//! # Filtering happens per subscriber, not centrally
//!
//! Each stream holds its own [`Subscribe`](crate::grpc::proto::Subscribe) and drops what it did not ask for.
//! The alternative — the ECS knowing who wants what — would put a registry of
//! live gRPC clients inside the scene, which is exactly the coupling the command
//! channel exists to avoid. The cost is that an event nobody wants is still
//! broadcast, which is a few bytes per click.

use tokio::sync::broadcast;

use crate::scene::SceneObject;
use bevy::prelude::*;

use super::proto::{ActorHandle, Event as ProtoEvent, EventKind, ObjectHandle, Vector3};

/// How many events a slow watcher may fall behind before it starts missing them.
///
/// Small on purpose. Events describe *what the user just did*, and a client far
/// enough behind to fill this is not going to be helped by a longer queue — it
/// wants the current state, which it can ask for. Bounded memory beats complete
/// history here.
const DEPTH: usize = 64;

/// One thing that happened, as the ECS reports it.
///
/// Handles rather than entities: an `Entity` is an ECS index that means nothing
/// outside this process, and everything on the wire speaks in handles.
#[derive(Debug, Clone)]
pub struct SceneEvent {
    pub kind: EventKind,
    pub object: u64,
    pub actor: u64,
    pub position: Option<Vec3>,
}

impl SceneEvent {
    pub(super) fn to_proto(&self) -> ProtoEvent {
        ProtoEvent {
            kind: self.kind as i32,
            object: Some(ObjectHandle { id: self.object }),
            actor: Some(ActorHandle { id: self.actor }),
            position: self.position.map(|at| Vector3 {
                x: at.x,
                y: at.y,
                z: at.z,
            }),
            // Never set yet. See the field's comment in scene.proto: it needs a
            // second ray cast for the triangle index, and then a walk back
            // through the filter graph to mean anything to a client.
            element: None,
        }
    }
}

/// The broadcast end held by the app.
///
/// A resource so any system can report something without knowing whether anyone
/// is listening — `send` on a channel with no receivers is not an error here,
/// and reporting into the void is the normal case.
#[derive(Resource, Clone)]
pub struct Events(broadcast::Sender<SceneEvent>);

impl Default for Events {
    fn default() -> Self {
        Self(broadcast::channel(DEPTH).0)
    }
}

impl Events {
    pub fn subscribe(&self) -> broadcast::Receiver<SceneEvent> {
        self.0.subscribe()
    }

    /// Reports an event. Deliberately ignores "nobody listening".
    pub fn send(&self, event: SceneEvent) {
        let _ = self.0.send(event);
    }
}

/// Turns picks into events for whoever is watching.
///
/// Separate from the UI's own `take_picks`, and reading the same messages: what
/// the interface does with a click and what a script is told about it are two
/// consumers of one fact, and neither should go through the other.
pub fn report_picks(
    mut picks: MessageReader<crate::scene::Picked>,
    events: Res<Events>,
    ids: Query<&crate::counter::UniqueID>,
    objects: Query<(), With<SceneObject>>,
) {
    for pick in picks.read() {
        // Both halves have to resolve, or the event would name a handle that
        // means nothing. An actor with no id is not a thing a client can act on.
        let (Ok(object), Ok(actor)) = (ids.get(pick.object), ids.get(pick.actor)) else {
            continue;
        };
        if !objects.contains(pick.object) {
            continue;
        }
        events.send(SceneEvent {
            kind: EventKind::Pick,
            object: object.0,
            actor: actor.0,
            position: pick.position,
        });
    }
}
