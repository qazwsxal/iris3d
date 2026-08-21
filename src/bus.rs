//! The channel commands arrive on, and the handle for submitting them.
//!
//! Generic over what it carries, because nothing here is specific to any one
//! kind of command: the scene and the filter graph each own a bus of their own,
//! and the two are the same machinery over different payloads.
//!
//! Crossbeam rather than a Bevy message, because the sending half is used from
//! threads Bevy does not own — the gRPC runtime, chiefly. A receiver is drained
//! by the crate that owns it, once per `Update`. Commands are applied in arrival
//! order, but because insertion and deletion go through Bevy's deferred
//! `Commands`, an entity becomes queryable on the tick *after* the one that
//! created it.

use bevy::prelude::*;
use bevy::winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent};
use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};

/// Both halves of a command channel, plus the means to wake the window.
#[derive(Resource)]
pub struct Bus<T: Send + Sync + 'static> {
    tx: Sender<T>,
    rx: Receiver<T>,
    wake: Option<EventLoopProxy<WinitUserEvent>>,
}

impl<T: Send + Sync + 'static> Bus<T> {
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
    /// producer that needs to talk to the owner of this bus.
    pub fn sender(&self) -> BusSender<T> {
        BusSender {
            commands: self.tx.clone(),
            wake: self.wake.clone(),
        }
    }

    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.rx.try_recv()
    }
}

impl<T: Send + Sync + 'static> Default for Bus<T> {
    fn default() -> Self {
        Self::new(None)
    }
}

/// A handle for submitting commands that also wakes the window.
///
/// The window only updates in response to events (see [`crate::redraw`]), and a
/// command arriving on another thread is not an event winit knows about.
/// Without the wake-up a scripted change would sit in the channel until the
/// idle tick came round, so waking is part of sending rather than something a
/// caller has to remember.
pub struct BusSender<T> {
    commands: Sender<T>,
    wake: Option<EventLoopProxy<WinitUserEvent>>,
}

// Derived `Clone` would demand `T: Clone`, which is wrong: a `Sender<T>` clones
// whatever `T` is, and a command carries a reply channel that cannot be cloned.
impl<T> Clone for BusSender<T> {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.clone(),
            wake: self.wake.clone(),
        }
    }
}

impl<T> BusSender<T> {
    /// Queues a command for the next tick, and makes sure a tick happens.
    pub fn send(&self, command: T) -> Result<(), Gone> {
        self.commands.send(command).map_err(|_| Gone)?;
        if let Some(wake) = &self.wake {
            // The only way this fails is the event loop having already exited,
            // which the caller finds out about when its reply never arrives.
            let _ = wake.send_event(WinitUserEvent::WakeUp);
        }
        Ok(())
    }
}

/// Nobody is draining this bus, so nothing more can be submitted. Only happens
/// once the app is on its way out. The command is dropped rather than handed
/// back, because there is nowhere else to put it.
#[derive(Debug)]
pub struct Gone;
