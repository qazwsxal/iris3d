//! The channel commands arrive on, and the handle for submitting them.
//!
//! Everything that changes the scene from outside the drain — the gRPC server,
//! the interface, a drag in the viewport — goes through here. It carries
//! [`SceneCommand`]s and nothing else, so it belongs to the scene rather than to
//! any one producer: gRPC is the busiest caller, not the owner.
//!
//! Crossbeam rather than a Bevy message, because the sending half is used from a
//! thread Bevy does not own. The receiver is drained by
//! [`apply_scene_commands`](super::apply_scene_commands) each `Update`. Commands
//! are applied in arrival order, but because insertion and deletion go through
//! Bevy's deferred `Commands`, an object becomes queryable on the tick *after*
//! the one that created it.

use bevy::prelude::*;
use bevy::winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent};
use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};

use super::SceneCommand;

/// Both halves of the command channel, plus the means to wake the window.
#[derive(Resource)]
pub struct CommandBus {
    tx: Sender<SceneCommand>,
    rx: Receiver<SceneCommand>,
    wake: Option<EventLoopProxy<WinitUserEvent>>,
}

impl CommandBus {
    pub fn new(wake: Option<EventLoopProxy<WinitUserEvent>>) -> Self {
        let (tx, rx) = unbounded();
        Self { tx, rx, wake }
    }

    /// Reads the winit proxy out of the world, if there is one.
    ///
    /// `None` without a window — a headless test, or before `WinitPlugin` has
    /// run. The bus still works; commands just wait for the next update instead
    /// of causing one.
    pub fn from_world(world: &World) -> Self {
        Self::new(
            world
                .get_resource::<EventLoopProxyWrapper>()
                .map(|proxy| (**proxy).clone()),
        )
    }

    /// A handle for submitting commands. Cheap to clone; hand one to every
    /// producer that needs to talk to the scene.
    pub fn sender(&self) -> SceneSender {
        SceneSender {
            commands: self.tx.clone(),
            wake: self.wake.clone(),
        }
    }

    pub fn try_recv(&self) -> Result<SceneCommand, TryRecvError> {
        self.rx.try_recv()
    }
}

impl Default for CommandBus {
    fn default() -> Self {
        Self::new(None)
    }
}

/// A handle for submitting [`SceneCommand`]s that also wakes the window.
///
/// The window only updates in response to events (see [`crate::redraw`]), and a
/// command arriving on another thread is not an event winit knows about.
/// Without the wake-up a scripted change would sit in the channel until the
/// idle tick came round, so waking is part of sending rather than something a
/// caller has to remember.
#[derive(Clone)]
pub struct SceneSender {
    commands: Sender<SceneCommand>,
    wake: Option<EventLoopProxy<WinitUserEvent>>,
}

impl SceneSender {
    /// Queues a command for the next tick, and makes sure a tick happens.
    pub fn send(&self, command: SceneCommand) -> Result<(), SceneGone> {
        self.commands.send(command).map_err(|_| SceneGone)?;
        if let Some(wake) = &self.wake {
            // The only way this fails is the event loop having already exited,
            // which the caller finds out about when its reply never arrives.
            let _ = wake.send_event(WinitUserEvent::WakeUp);
        }
        Ok(())
    }
}

/// The scene is no longer draining commands, so nothing more can be submitted.
/// Only happens once the app is on its way out. The command is dropped rather
/// than handed back, because there is nowhere else to put it.
#[derive(Debug)]
pub struct SceneGone;
