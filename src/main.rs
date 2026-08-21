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
//! - **Data** is arrays, held flat in `crate::data::DataStore` and referred to
//!   by handle. An array belongs to no object, so one array can feed several
//!   representations and one representation can read arrays that arrived
//!   separately.
//! - An **object** ([`crate::scene::SceneObject`]) is a place in the tree and a
//!   name. It holds no data.
//! - An **actor** is one way of drawing something. It binds arrays to the
//!   inputs its kind declares, and is drawn under the objects it is placed
//!   under.
//! - A **filter** ([`filter`]) derives arrays from arrays and draws nothing.
//!
//! Filters are what keep generating geometry separate from displaying it: `N`
//! generators and `M` displays cost `N + M` implementations rather than
//! `N * M`. See `docs/design/filters.md` for the full argument.
//!
//! # The layers
//!
//! Work flows one way. Each module may only name the ones above it in this
//! list, and `.github/workflows/ci.yml` checks that on every push — the check
//! is a grep over `crate::` imports, which is cheap enough to be worth more
//! than the ceremony of a crate per layer.
//!
//! ```text
//!   grpc ──SceneCommand──▶ scene ──▶ filter ──▶ draw ──▶ screen
//!    ▲                       │                            │
//!    └───── watch events ────┴────────── view ◀───────────┘
//! ```
//!
//! Bottom to top:
//!
//! - [`bus`], [`counter`], [`redraw`] — the plumbing every layer uses and none
//!   owns: the channel commands arrive on, the handles clients name things by,
//!   and when the app draws at all. None of them knows what an array is.
//! - [`data`] — arrays, dtypes, the store, and the periodic table.
//! - [`model`] — what a client and the server agree about: declared parameters,
//!   the bindings that satisfy them, and the ways a request can be refused.
//!   Both the scene and the filter graph raise those, which is why they are
//!   here rather than in either.
//! - [`scene`] — the tree, the actor registry, and the scene commands. Draws
//!   nothing and knows nothing about gRPC.
//! - [`filter`] — derives arrays. Knows nothing about pipelines, and runs off
//!   the main thread. Owns its own commands, because a filter has no place in
//!   the tree.
//! - [`draw`] — rendering backends. One whole pathway is chosen at launch;
//!   `default` is the only one built.
//! - [`view`] — the viewport, picking, overlays and the egui interface. Nothing
//!   below it knows it exists; a build with no interface still draws.
//! - [`grpc`] — the tonic server. An adapter onto the two command buses and
//!   nothing more.
//!
//! # Where to start reading
//!
//! To follow one request end to end: `crate::grpc::scene_service` converts it,
//! [`crate::scene::SceneCommand`] names it, and
//! [`crate::scene::apply_scene_commands`] applies it.
//!
//! To add something: `docs/adding-a-filter.md` and
//! `docs/adding-an-actor-kind.md`. [`crate::filter::colormap`] is the smallest
//! filter worth copying; [`crate::draw::default::points`] is the smallest actor
//! kind.

use bevy::prelude::*;
use clap::Parser;

mod bus;
mod capture;
mod cli;
mod counter;
mod data;
mod draw;
mod filter;
mod grpc;
mod model;
mod redraw;
mod scene;
mod view;

use cli::Cli;
use counter::CounterPlugin;
use draw::DrawPlugin;
use filter::FilterPlugin;
use grpc::GrpcPlugin;
use redraw::RedrawPlugin;
use scene::ScenePlugin;
use view::{UiPlugin, ViewportPlugin};
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
