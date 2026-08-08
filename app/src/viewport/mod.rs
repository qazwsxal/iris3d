//! Camera, lighting and navigation.
//!
//! Backend-agnostic on purpose: whichever way objects end up being drawn, they
//! need a view to be drawn into. Nothing here knows about representations.

use bevy::camera::primitives::Aabb;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use std::time::Duration;

pub mod overlays;

pub use overlays::OverlaySettings;

pub struct ViewportPlugin;

impl Plugin for ViewportPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenshotRequest>()
            .init_resource::<FrameRequest>()
            .init_resource::<PointerCaptured>()
            .init_resource::<OverlaySettings>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    orbit_controls,
                    frame_content,
                    screenshot,
                    overlays::draw_overlays,
                ),
            );
    }
}

/// An orbit camera looking at `focus` from `distance` away.
#[derive(Component, Debug)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            distance: 12.0,
            yaw: 0.6,
            pitch: 0.5,
        }
    }
}

impl OrbitCamera {
    fn transform(&self) -> Transform {
        let offset = Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        );
        Transform::from_translation(self.focus + offset).looking_at(self.focus, Vec3::Y)
    }
}

fn setup(mut commands: Commands) {
    let camera = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        camera.transform(),
        camera,
        // Ambient light is per-view rather than a global resource.
        AmbientLight {
            color: Color::WHITE,
            brightness: 200.0,
            ..default()
        },
    ));

    // A key light with a dimmer fill from behind, so unlit faces do not go
    // completely flat. Scientific data has no artistic lighting to fall back
    // on — shape has to read from shading alone.
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(6.0, 10.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 2_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-8.0, -4.0, -6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Set when an overlay wants the mouse, so dragging a panel does not also
/// orbit the camera underneath it. Kept as a plain flag rather than an egui
/// dependency, so the viewport stays independent of whichever UI is on top.
#[derive(Resource, Default)]
pub struct PointerCaptured(pub bool);

/// Left drag orbits, right or middle drag pans, wheel zooms.
fn orbit_controls(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    captured: Res<PointerCaptured>,
    mut camera: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    if captured.0 {
        return;
    }
    let Ok((mut orbit, mut transform)) = camera.single_mut() else {
        return;
    };

    let mut changed = false;

    if buttons.pressed(MouseButton::Left) && motion.delta != Vec2::ZERO {
        orbit.yaw -= motion.delta.x * 0.005;
        orbit.pitch = (orbit.pitch + motion.delta.y * 0.005).clamp(-1.54, 1.54);
        changed = true;
    }

    if (buttons.pressed(MouseButton::Right) || buttons.pressed(MouseButton::Middle))
        && motion.delta != Vec2::ZERO
    {
        // Pan in the camera's own plane, scaled by distance so the drag feels
        // the same whether zoomed in or out.
        let distance = orbit.distance;
        let right = transform.right() * -motion.delta.x;
        let up = transform.up() * motion.delta.y;
        orbit.focus += (right + up) * distance * 0.002;
        changed = true;
    }

    if scroll.delta.y != 0.0 {
        let step = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y * 0.1,
            MouseScrollUnit::Pixel => scroll.delta.y * 0.005,
        };
        orbit.distance = (orbit.distance * (1.0 - step)).clamp(0.05, 10_000.0);
        changed = true;
    }

    if changed {
        *transform = orbit.transform();
    }
}

/// A request to reframe the view, raised by the UI.
#[derive(Resource, Default)]
pub struct FrameRequest(pub Option<FrameTarget>);

pub enum FrameTarget {
    /// Fit everything currently drawn.
    All,
    /// Fit one object, including whatever its descendants draw.
    Subtree(Entity),
}

/// Frames the view when new geometry appears, or when the UI asks.
///
/// Driven off `Aabb`, which Bevy computes for any mesh, so this works for any
/// rendering backend that produces meshes without knowing anything about it.
fn frame_content(
    time: Res<Time>,
    changed: Query<(), (With<Aabb>, Or<(Added<Aabb>, Changed<GlobalTransform>)>)>,
    bounds: Query<(&Aabb, &GlobalTransform)>,
    children: Query<&Children>,
    mut request: ResMut<FrameRequest>,
    mut camera: Query<(&mut OrbitCamera, &mut Transform, &Projection, &Camera)>,
    mut settle: Local<Option<Timer>>,
) {
    let target = request.0.take();

    // Wait for the scene to stop changing before fitting.
    //
    // A client typically uploads an object, parents it, then positions it —
    // three separate calls landing over several frames. Fitting the moment
    // geometry appears would use its pre-transform position, and the last
    // object uploaded would never be accounted for at all. Watching transforms
    // as well as new meshes, and only fitting once both have been quiet for a
    // moment, gets the final arrangement. It also gives the UI a chance to
    // inset the camera viewport, so the aspect ratio is the real one.
    if !changed.is_empty() {
        *settle = Some(Timer::from_seconds(0.25, TimerMode::Once));
    }
    let settled = settle
        .as_mut()
        .is_some_and(|timer| timer.tick(time.delta()).just_finished());
    if settled {
        *settle = None;
    }
    if target.is_none() && !settled {
        return;
    }

    let Ok((mut orbit, mut transform, projection, camera_view)) = camera.single_mut() else {
        return;
    };

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let include = |entity: Entity, min: &mut Vec3, max: &mut Vec3| {
        if let Ok((aabb, global)) = bounds.get(entity) {
            let centre = global.transform_point(Vec3::from(aabb.center));
            let extent = (global.affine().matrix3 * Vec3::from(aabb.half_extents)).abs();
            *min = min.min(centre - extent);
            *max = max.max(centre + extent);
        }
    };

    match target {
        Some(FrameTarget::Subtree(root)) => {
            include(root, &mut min, &mut max);
            // Descendants hold the geometry — an object entity itself has no
            // mesh, its representation children do.
            let mut stack = vec![root];
            while let Some(entity) = stack.pop() {
                if let Ok(kids) = children.get(entity) {
                    for kid in kids.iter() {
                        include(kid, &mut min, &mut max);
                        stack.push(kid);
                    }
                }
            }
        }
        _ => {
            for (aabb, global) in &bounds {
                let centre = global.transform_point(Vec3::from(aabb.center));
                let extent = (global.affine().matrix3 * Vec3::from(aabb.half_extents)).abs();
                min = min.min(centre - extent);
                max = max.max(centre + extent);
            }
        }
    }

    if !min.is_finite() || !max.is_finite() {
        return;
    }

    orbit.focus = (min + max) * 0.5;

    // Fit the bounding sphere to whichever field of view is tighter. The UI
    // insets the camera to the space between panels, so the viewport is often
    // much wider than it is tall — fitting only the vertical FOV would let
    // content run off the sides.
    let radius = ((max - min).length() * 0.5).max(0.001);
    let half_vertical = match projection {
        Projection::Perspective(perspective) => perspective.fov * 0.5,
        _ => std::f32::consts::FRAC_PI_8,
    };
    // Taken from the camera rather than the projection: the projection's
    // aspect_ratio lags a frame behind a viewport change.
    let aspect = camera_view
        .logical_viewport_size()
        .filter(|size| size.y > 0.0)
        .map(|size| size.x / size.y)
        .unwrap_or(1.0);
    let half_horizontal = (half_vertical.tan() * aspect.max(0.01)).atan();
    let half_angle = half_vertical.min(half_horizontal).max(0.01);
    orbit.distance = (radius / half_angle.tan()) * 1.15;
    *transform = orbit.transform();
    info!(
        "viewport: framed bounds {:.2} .. {:.2} (radius {:.2})",
        min, max, radius
    );
}

/// Saves a PNG of the window.
///
/// Press F12, or set `IRIS3D_SCREENSHOT=<path>` to capture automatically after
/// `IRIS3D_SCREENSHOT_DELAY` seconds (default 15) — enough to script a render
/// without a person at the keyboard. `IRIS3D_SCREENSHOT_EXIT=1` closes the app
/// once the file is on disk.
#[derive(Resource)]
struct ScreenshotRequest {
    path: Option<String>,
    countdown: Timer,
    exit_after: bool,
    fired: bool,
    exit_at: Option<Timer>,
}

impl Default for ScreenshotRequest {
    fn default() -> Self {
        let delay = std::env::var("IRIS3D_SCREENSHOT_DELAY")
            .ok()
            .and_then(|raw| raw.parse::<f32>().ok())
            .unwrap_or(15.0);
        Self {
            path: std::env::var("IRIS3D_SCREENSHOT").ok(),
            countdown: Timer::new(Duration::from_secs_f32(delay), TimerMode::Once),
            exit_after: std::env::var("IRIS3D_SCREENSHOT_EXIT").is_ok(),
            fired: false,
            exit_at: None,
        }
    }
}

fn screenshot(
    mut commands: Commands,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut request: ResMut<ScreenshotRequest>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(timer) = request.exit_at.as_mut() {
        if timer.tick(time.delta()).just_finished() {
            exit.write(AppExit::Success);
        }
        return;
    }

    let automatic = request.path.is_some()
        && !request.fired
        && request.countdown.tick(time.delta()).just_finished();

    let path = if automatic {
        request.path.clone()
    } else if keys.just_pressed(KeyCode::F12) {
        Some(format!(
            "iris3d-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default()
        ))
    } else {
        None
    };

    let Some(path) = path else { return };

    request.fired = true;
    info!("viewport: saving screenshot to {path}");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));

    if automatic && request.exit_after {
        request.exit_at = Some(Timer::new(Duration::from_secs(2), TimerMode::Once));
    }
}
