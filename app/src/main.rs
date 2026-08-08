use bevy::prelude::*;

mod counter;
mod draw;
mod grpc;
mod scene;
mod ui;
mod viewport;

use counter::CounterPlugin;
use draw::DrawPlugin;
use grpc::GrpcPlugin;
use scene::ScenePlugin;
use ui::UiPlugin;
use viewport::ViewportPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(CounterPlugin)
        .add_plugins(GrpcPlugin::default())
        .add_plugins(ScenePlugin)
        .add_plugins(ViewportPlugin)
        .add_plugins(DrawPlugin)
        .add_plugins(UiPlugin)
        .run();
}
