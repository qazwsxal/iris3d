//! Does `bevy_solari` draw anything at all on this machine?
//!
//! A sanity check that shares nothing with iris3d beyond the dependency: a
//! cube, a ground plane, one directional light, and a camera set up exactly as
//! Bevy's own `solari` example sets one up. No iris3d plugins, no actors, no
//! reactive redraw.
//!
//! If this renders, `bevy_solari` works here and the fault is in iris3d's
//! integration. If this is black too, the fault is the driver, the adapter, or
//! a missing denoiser, and no amount of work on the backend will fix it.
//!
//! Run with:
//!   cargo run --example solari_smoke -- out.png

use bevy::camera::CameraMainTextureUsages;
use bevy::light::light_consts;
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins};

/// Long enough for the acceleration structures to build and for a raytraced
/// image to accumulate out of the initial noise.
const CAPTURE_AT: u32 = 600;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "solari_smoke.png".to_string());

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SolariPlugins)
        .insert_resource(Output(path))
        .add_systems(Startup, setup)
        .add_systems(Update, capture)
        .run();
}

#[derive(Resource)]
struct Output(String);

/// Everything Solari needs of a mesh: triangles, normals, UVs, tangents and
/// 32-bit indices. `generate_tangents` needs the UVs and normals to be there
/// already, which the primitive meshes provide.
fn raytraceable(mut mesh: Mesh) -> Mesh {
    mesh.generate_tangents()
        .expect("a primitive mesh should take tangents");
    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cube = meshes.add(raytraceable(Cuboid::new(1.0, 1.0, 1.0).into()));
    let ground = meshes.add(raytraceable(
        Plane3d::default().mesh().size(10.0, 10.0).build(),
    ));
    let red = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.2, 0.2),
        perceptual_roughness: 0.6,
        ..default()
    });
    let grey = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.7),
        perceptual_roughness: 0.9,
        ..default()
    });

    commands.spawn((
        Mesh3d(cube.clone()),
        RaytracingMesh3d(cube),
        MeshMaterial3d(red),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
    commands.spawn((
        Mesh3d(ground.clone()),
        RaytracingMesh3d(ground),
        MeshMaterial3d(grey),
        Transform::default(),
    ));

    // Bevy's example uses full daylight and no shadow maps, because Solari
    // replaces shadow mapping.
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(4.0, 3.0, 6.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        // The two the example calls non-negotiable, plus the lighting itself.
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariLighting::default(),
    ));
}

fn capture(
    mut commands: Commands,
    output: Res<Output>,
    mut quit: MessageWriter<AppExit>,
    mut frame: Local<u32>,
) {
    *frame += 1;
    if *frame == CAPTURE_AT {
        info!("solari_smoke: capturing to {}", output.0);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(output.0.clone()));
    }
    if *frame >= CAPTURE_AT + 60 {
        quit.write(AppExit::Success);
    }
}
