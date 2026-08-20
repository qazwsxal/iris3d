//! iris3d — a scriptable 3D visualisation suite for scientific data.
//!
//! The application opens a window, holds a scene, and waits to be told what to
//! put in it over gRPC. The wire contract in `proto/iris3d/v1/scene.proto` is
//! the real interface; this workspace implements it. See `README.md` for how to
//! run it and drive it from Python.
//!
//! **This crate is wiring only.** It parses the command line and adds the
//! plugins in order. Everything else is in the crates below it.
//!
//! # The model
//!
//! Four things, deliberately kept apart:
//!
//! - **Data** is arrays, held flat in `iris3d_data::DataStore` and referred to
//!   by handle. An array belongs to no object, so one array can feed several
//!   representations and one representation can read arrays that arrived
//!   separately.
//! - An **object** ([`iris3d_scene::SceneObject`]) is a place in the tree and a
//!   name. It holds no data.
//! - An **actor** is one way of drawing something. It binds arrays to the
//!   inputs its kind declares, and is drawn under the objects it is placed
//!   under.
//! - A **filter** ([`iris3d_filter`]) derives arrays from arrays and draws
//!   nothing.
//!
//! Filters are what keep generating geometry separate from displaying it: `N`
//! generators and `M` displays cost `N + M` implementations rather than
//! `N * M`. See `docs/design/filters.md` for the full argument.
//!
//! # The layers
//!
//! Work flows one way, and the crate graph enforces it — a crate may only name
//! the ones beneath it, so a re-formed cycle is a build failure rather than a
//! comment that goes stale.
//!
//! ```text
//!   grpc ──SceneCommand──▶ scene ──▶ filter ──▶ draw ──▶ screen
//!    ▲                       │                            │
//!    └───── watch events ────┴────────── view ◀───────────┘
//! ```
//!
//! Bottom to top:
//!
//! - [`iris3d_core`] — the command bus, the handle counter, the redraw policy.
//!   Plumbing every layer needs and none owns.
//! - `iris3d_data` — arrays, dtypes, the store, and the periodic table.
//! - `iris3d_model` — what a client and the server agree about: declared
//!   parameters, the bindings that satisfy them, and the ways a request can be
//!   refused.
//! - [`iris3d_scene`] — the tree, the actor registry, and the scene commands.
//!   Draws nothing and knows nothing about gRPC.
//! - [`iris3d_filter`] — derives arrays. Knows nothing about pipelines, and runs
//!   off the main thread. Owns its own commands, because a filter has no place
//!   in the tree.
//! - [`iris3d_draw`] — rendering backends. One whole pathway is chosen at
//!   launch; `default` is the only one built.
//! - [`iris3d_view`] — the viewport, picking, overlays and the egui interface.
//!   Nothing below it knows it exists; a build with no interface still draws.
//! - [`iris3d_grpc`] — the tonic server. An adapter onto the two command buses
//!   and nothing more.
//!
//! # Where to start reading
//!
//! To follow one request end to end: `iris3d_grpc::scene_service` converts it,
//! [`iris3d_scene::SceneCommand`] names it, and
//! [`iris3d_scene::apply_scene_commands`] applies it.
//!
//! To add something: `docs/adding-a-filter.md` and
//! `docs/adding-an-actor-kind.md`. `iris3d_filter::colormap` is the smallest
//! filter worth copying; `iris3d_draw::default::points` is the smallest actor
//! kind.

use bevy::prelude::*;
use clap::Parser;

mod capture;
mod cli;

use cli::Cli;
use iris3d_core::{CounterPlugin, RedrawPlugin};
use iris3d_draw::DrawPlugin;
use iris3d_filter::FilterPlugin;
use iris3d_grpc::GrpcPlugin;
use iris3d_scene::ScenePlugin;
use iris3d_view::{UiPlugin, ViewportPlugin};
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
