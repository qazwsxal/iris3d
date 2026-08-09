use bevy::prelude::*;

mod counter;
mod draw;
mod grpc;
mod redraw;
mod scene;
mod ui;
mod viewport;

use counter::CounterPlugin;
use draw::DrawPlugin;
use grpc::GrpcPlugin;
use redraw::RedrawPlugin;
use scene::ScenePlugin;
use ui::UiPlugin;
use viewport::ViewportPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // After `DefaultPlugins`: both of these need what `WinitPlugin` sets up.
        .add_plugins(RedrawPlugin)
        .add_plugins(CounterPlugin)
        .add_plugins(GrpcPlugin::default())
        .add_plugins(ScenePlugin)
        .add_plugins(ViewportPlugin)
        .add_plugins(DrawPlugin)
        .add_plugins(UiPlugin)
        .run();
}
