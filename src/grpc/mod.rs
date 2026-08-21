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
use std::net::SocketAddr;
use tonic::transport::Server;

use crate::bus::BusSender;
use crate::filter::FilterCommand;
use crate::scene::SceneCommand;

pub mod convert;
pub mod scene_service;
pub mod upload;
pub mod watch;

/// Generated types for `proto/iris3d/v1/scene.proto`.
pub mod proto {
    tonic::include_proto!("iris3d.v1");
}

use proto::scene_service_server::SceneServiceServer;

/// Largest chunk the server will decode. Clients are advised in the proto to
/// stay in the 1-4 MiB range; this leaves headroom above that.
const MAX_DECODING_MESSAGE_SIZE: usize = 8 * 1024 * 1024;

/// Serves the gRPC surface, submitting onto the scene's command bus.
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
        // The buses belong to the scene and the filter graph, so both plugins
        // have to be added first. gRPC is one producer onto each and holds a
        // sender like any other.
        let scene = app
            .world()
            .get_resource::<crate::scene::CommandBus>()
            .expect("ScenePlugin must be added before GrpcPlugin")
            .sender();
        let filters = app
            .world()
            .get_resource::<crate::filter::FilterBus>()
            .expect("FilterPlugin must be added before GrpcPlugin")
            .sender();
        // Built here rather than inside the server thread: the service needs
        // the same channel the ECS subscribes watchers to.
        let events = watch::Events::default();
        spawn_server(self.addr, scene, filters, events.clone());
        app.insert_resource(events)
            .add_systems(Update, watch::report_picks);
    }
}

/// Starts the tonic server on a dedicated thread.
///
/// The server only needs the command channel, so it can come up before the
/// Bevy app starts ticking; requests simply queue until the scene drains them.
fn spawn_server(
    addr: SocketAddr,
    scene: BusSender<SceneCommand>,
    filters: BusSender<FilterCommand>,
    events: watch::Events,
) {
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
                let service = scene_service::SceneBridgeService::new(scene, filters, events);
                let service = SceneServiceServer::new(service)
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
