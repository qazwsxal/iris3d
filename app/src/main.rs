use bevy::prelude::*;
use clap::Parser;

mod capture;
mod cli;
mod counter;
mod draw;
mod filter;
mod grpc;
mod redraw;
mod scene;
mod ui;
mod viewport;

use cli::Cli;
use counter::CounterPlugin;
use draw::DrawPlugin;
use filter::FilterPlugin;
use grpc::GrpcPlugin;
use redraw::RedrawPlugin;
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
        .add_plugins(GrpcPlugin { addr: cli.listen })
        .add_plugins(ScenePlugin)
        .add_plugins(ViewportPlugin)
        // Above the backend: filters derive arrays, and know nothing about how
        // anything is drawn.
        .add_plugins(FilterPlugin)
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
