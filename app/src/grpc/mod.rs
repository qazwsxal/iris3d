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
use crossbeam_channel::{Receiver, Sender, TryRecvError, unbounded};
use std::net::SocketAddr;
use tonic::transport::Server;

use crate::scene::SceneCommand;

pub mod scene_service;

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
}

impl GrpcBridge {
    pub fn new() -> Self {
        let (tx, rx) = unbounded();
        Self { tx, rx }
    }

    /// A handle for submitting commands. Cheap to clone; hand one to every
    /// producer that needs to talk to the scene.
    pub fn sender(&self) -> Sender<SceneCommand> {
        self.tx.clone()
    }

    pub fn try_recv(&self) -> Result<SceneCommand, TryRecvError> {
        self.rx.try_recv()
    }
}

impl Default for GrpcBridge {
    fn default() -> Self {
        Self::new()
    }
}

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
        let bridge = GrpcBridge::new();
        spawn_server(self.addr, bridge.sender());
        app.insert_resource(bridge);
    }
}

/// Starts the tonic server on a dedicated thread.
///
/// The server only needs the command channel, so it can come up before the
/// Bevy app starts ticking; requests simply queue until the scene drains them.
fn spawn_server(addr: SocketAddr, commands: Sender<SceneCommand>) {
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
                let scene = scene_service::SceneBridgeService::new(commands);
                let service =
                    SceneServiceServer::new(scene).max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE);

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
