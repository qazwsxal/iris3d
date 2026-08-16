//! The gRPC control surface.
//!
//! iris3d is scriptable from any language, so the wire contract in `proto/` is
//! the real interface and this module is only an adapter onto it. Everything
//! here converts protobuf messages into [`SceneCommand`]s and back; the scene
//! itself does not depend on gRPC.
//!
//! Threading: Bevy owns the main thread, so the tonic server runs on its own
//! thread with its own tokio runtime. The two sides never touch each other's
//! state — they communicate over a crossbeam channel, and bulk transfers are
//! fully assembled on the tokio side before the ECS ever sees them.

use bevy::prelude::*;
use bevy::winit::{EventLoopProxy, EventLoopProxyWrapper, WinitUserEvent};
use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};
use std::net::SocketAddr;
use tonic::transport::Server;

use crate::scene::SceneCommand;

pub mod scene_service;
pub mod watch;

/// Generated types for `proto/iris3d/v1/scene.proto`.
pub mod proto {
    tonic::include_proto!("iris3d.v1");
}

use proto::scene_service_server::SceneServiceServer;

/// Largest chunk the server will decode. Clients are advised in the proto to
/// stay in the 1-4 MiB range; this leaves headroom above that.
const MAX_DECODING_MESSAGE_SIZE: usize = 8 * 1024 * 1024;

/// Channel carrying work from the gRPC runtime into the ECS.
///
/// The receiver is drained by the scene each `Update`. Commands are applied in
/// arrival order, but because insertion and deletion go through Bevy's deferred
/// `Commands`, an object becomes queryable on the tick *after* the one that
/// created it.
#[derive(Resource)]
pub struct GrpcBridge {
    tx: Sender<SceneCommand>,
    rx: Receiver<SceneCommand>,
    wake: Option<EventLoopProxy<WinitUserEvent>>,
}

impl GrpcBridge {
    pub fn new(wake: Option<EventLoopProxy<WinitUserEvent>>) -> Self {
        let (tx, rx) = unbounded();
        Self { tx, rx, wake }
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

impl Default for GrpcBridge {
    fn default() -> Self {
        Self::new(None)
    }
}

/// A handle for submitting [`SceneCommand`]s that also wakes the window.
///
/// The window only updates in response to events (see [`crate::redraw`]), and a
/// command arriving on the gRPC thread is not an event winit knows about.
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

/// Serves the gRPC surface and inserts the [`GrpcBridge`] the scene drains.
pub struct GrpcPlugin {
    pub addr: SocketAddr,
}

impl Default for GrpcPlugin {
    /// Loopback only. iris3d is a desktop application being scripted by
    /// processes on the same machine; binding a wider interface should be a
    /// deliberate choice, not the default.
    fn default() -> Self {
        Self {
            addr: "[::1]:50051".parse().expect("valid default address"),
        }
    }
}

impl Plugin for GrpcPlugin {
    fn build(&self, app: &mut App) {
        // Added by `WinitPlugin`, so this plugin has to come after
        // `DefaultPlugins`. Without the proxy the server still works; commands
        // just wait for the next update instead of causing one.
        let wake = app
            .world()
            .get_resource::<EventLoopProxyWrapper>()
            .map(|proxy| (**proxy).clone());
        if wake.is_none() {
            warn!("grpc: no event loop to wake; commands will wait for the next update");
        }

        let bridge = GrpcBridge::new(wake);
        // Built here rather than inside the server thread: the ECS needs the
        // sending half as a resource, and the service needs the same channel to
        // subscribe watchers to.
        let events = watch::Events::default();
        spawn_server(self.addr, bridge.sender(), events.clone());
        app.insert_resource(bridge)
            .insert_resource(events)
            .add_systems(Update, watch::report_picks);
    }
}

/// Starts the tonic server on a dedicated thread.
///
/// The server only needs the command channel, so it can come up before the
/// Bevy app starts ticking; requests simply queue until the scene drains them.
fn spawn_server(addr: SocketAddr, commands: SceneSender, events: watch::Events) {
    let spawned = std::thread::Builder::new()
        .name("iris3d-grpc".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("iris3d-grpc-worker")
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    error!("grpc: could not start tokio runtime: {err}");
                    return;
                }
            };

            runtime.block_on(async move {
                let scene = scene_service::SceneBridgeService::new(commands, events);
                let service = SceneServiceServer::new(scene)
                    .max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE);

                info!("grpc: listening on {addr}");
                if let Err(err) = Server::builder().add_service(service).serve(addr).await {
                    error!("grpc: server stopped: {err}");
                }
            });
        });

    if let Err(err) = spawned {
        error!("grpc: could not spawn server thread: {err}");
    }
}
