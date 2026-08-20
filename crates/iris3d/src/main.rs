//! iris3d — a scriptable 3D visualisation suite for scientific data.
//!
//! The application opens a window, holds a scene, and waits to be told what to
//! put in it over gRPC. The wire contract in `proto/iris3d/v1/scene.proto` is
//! the real interface; this crate implements it. See `README.md` for how to run
//! it and drive it from Python.
//!
//! # The model
//!
//! Four things, deliberately kept apart:
//!
//! - **Data** is arrays, held flat in [`scene::DataStore`] and referred to by
//!   handle. An array belongs to no object, so one array can feed several
//!   representations and one representation can read arrays that arrived
//!   separately.
//! - An **object** ([`scene::SceneObject`]) is a place in the tree and a name.
//!   It holds no data.
//! - An **actor** is one way of drawing something. It binds arrays to the
//!   inputs its kind declares, and is drawn under the objects it is placed
//!   under.
//! - A **filter** ([`filter`]) derives arrays from arrays and draws nothing.
//!
//! Filters are what keep generating geometry separate from displaying it: `N`
//! generators and `M` displays cost `N + M` implementations rather than
//! `N * M`. See [`filter`] for the full argument.
//!
//! # The layers
//!
//! Each module below is a Bevy plugin, added in [`main`]. Work flows one way:
//!
//! ```text
//!   grpc  ──SceneCommand──▶  scene  ──▶  filter  ──▶  draw  ──▶  screen
//!    ▲                         │                                   │
//!    └────── watch events ─────┴──────── ui, viewport ◀────────────┘
//! ```
//!
//! - [`scene`] — the tree, the data store, the command queue and the actor
//!   registry. Draws nothing.
//! - [`filter`] — derives arrays. Knows nothing about pipelines. Runs off the
//!   main thread.
//! - [`draw`] — rendering backends. One whole pathway is chosen at launch;
//!   [`draw::default`](mod@crate::draw::default) is the only one built.
//! - [`ui`] — the egui panel and the node graph. Reads the world and emits
//!   actions rather than mutating directly.
//! - [`viewport`] — camera, picking, transform handles and overlays.
//! - [`grpc`] — the tonic server. An adapter onto [`scene::SceneCommand`] and
//!   nothing more.
//! - [`counter`] — allocates the handles clients speak in.
//! - [`redraw`] — keeps the window drawing only when there is a reason to.
//!
//! A module never depends on one below it in that list without a stated reason.
//!
//! # Where to start reading
//!
//! To follow one request end to end: [`grpc::scene_service`] converts it,
//! [`scene::SceneCommand`] names it, and [`scene::apply_scene_commands`] applies
//! it.
//!
//! To add something: `docs/adding-a-filter.md` and
//! `docs/adding-an-actor-kind.md`. [`filter::colormap`] is the smallest filter
//! worth copying; `draw::default::points` is the smallest actor kind.

use bevy::prelude::*;
use clap::Parser;

mod capture;
mod cli;
mod draw;
mod filter;
mod grpc;
mod scene;
mod select;
mod ui;
mod viewport;

use iris3d_core::{CounterPlugin, RedrawPlugin};

use cli::Cli;
use draw::DrawPlugin;
use filter::FilterPlugin;
use grpc::GrpcPlugin;
use scene::ScenePlugin;
use ui::UiPlugin;
use viewport::ViewportPlugin;

fn main() {
    // Before the app is built, so `--help` prints and exits without opening a
    // window.
    let cli = Cli::parse();

    let mut app = App::new();
    app.add_plugins(DefaultPlugins)
        // After `DefaultPlugins`: both of these need what `WinitPlugin` sets up.
        .add_plugins(RedrawPlugin)
        .add_plugins(CounterPlugin)
        // Both before `GrpcPlugin`: each owns a command bus, and gRPC takes a
        // sender from each like every other producer.
        .add_plugins(ScenePlugin)
        // Above the backend: filters derive arrays, and know nothing about how
        // anything is drawn.
        .add_plugins(FilterPlugin)
        .add_plugins(GrpcPlugin { addr: cli.listen })
        .add_plugins(ViewportPlugin)
        // Exactly one rendering pathway. See `draw`.
        .add_plugins(DrawPlugin)
        .add_plugins(UiPlugin);

    if let Some(path) = cli.screenshot {
        app.add_plugins(capture::CapturePlugin {
            path,
            after: cli.screenshot_after,
        });
    }

    app.run();
}
