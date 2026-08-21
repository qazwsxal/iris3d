//! Dragging the selected object around, with handles you can actually grab.
//!
//! The overlays next door draw a bounding box and a set of axes for the selected
//! object, and both are read-only. This makes the axes *handles*: grab one and
//! the object moves along it, turns about it, or grows along it.
//!
//! # Why this is hand-rolled
//!
//! `transform-gizmo-bevy` is the obvious dependency and does not fit: as of
//! 0.10.0, published 2026-08-08, every one of its `bevy_*` dependencies is
//! `^0.18` and this project is on 0.19. That was the reason recorded when the
//! read-only overlays were written, and re-checking it before starting was worth
//! the minute it took — the crate had shipped twice since, and neither release
//! moved. Revisit when it does; the interaction here is small enough to throw
//! away.
//!
//! # The write path already existed
//!
//! Nothing here writes a `Transform` directly. A drag sends
//! [`SceneCommand::SetTransform`] down the same channel a script uses, so
//! dragging an object and calling `set_transform` from Python are the same
//! operation, validated once. The interface has always taken this path — the
//! object tree's Delete does too — so the gizmo needed no new authority.
//!
//! # Three problems that are not the maths
//!
//! - **Left-drag already orbits the camera.** A grabbed handle has to win, and
//!   the claim has to be made on *press* and held for the whole gesture, or
//!   leaving the handle mid-drag hands the rest of the motion to the camera.
//!   [`GizmoDrag`] is what `orbit_controls` checks.
//! - **The app renders reactively.** Without nudging [`KeepAwake`] a drag
//!   advances a frame at a time as other things happen to wake it.
//! - **The 3D camera is inset** to whatever the panels leave, so a pointer
//!   position has to be taken against `logical_viewport_rect`, not the window.
//!   `Camera::viewport_to_world` handles this, given the right rect-relative
//!   position — which is what `Camera::viewport_to_world` already expects.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::counter::UniqueID;
use crate::redraw::KeepAwake;
use crate::scene::CommandBus;
use crate::scene::{SceneCommand, SceneObject};
use crate::view::select::Selection;

use super::OrbitCamera;

/// How the handles move the object.
///
/// A resource owned by the viewport, which is what acts on it. The interface
/// writes it — that is the only thing the radio buttons do — and reads it back
/// to show which one is active.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

/// Whether the gizmo is being dragged, and along which axis.
///
/// Read by [`orbit_controls`](super::orbit_controls): while this holds a grab,
/// the camera ignores the left button entirely. Claimed on press and released
/// on release, so the gesture belongs to whatever it started on.
#[derive(Resource, Default)]
pub struct GizmoDrag(pub Option<Grab>);

/// One in-progress drag.
#[derive(Debug, Clone, Copy)]
pub struct Grab {
    /// The object being moved, by handle — the name a command speaks in.
    pub object: u64,
    /// World-space axis, one of X, Y or Z.
    pub axis: Vec3,
    /// Where the axis line sits, **fixed for the whole gesture**.
    ///
    /// Not re-read from the object each frame, and that is not an optimisation.
    /// The handles are anchored to the middle of what is drawn, so moving the
    /// object moves the anchor — and measuring this frame's pointer against an
    /// anchor that already absorbed last frame's motion subtracts the movement
    /// from itself. The object judders back and forth under the pointer instead
    /// of following it.
    ///
    /// Pinning the line makes the drag a pure function of where the pointer is,
    /// so it cannot feed back into itself.
    pub origin: Vec3,
    /// Where along the axis the pointer was when the drag began, so the object
    /// does not jump to the pointer on the first frame.
    pub offset: f32,
    /// The transform the object had when the drag started. Every frame's value
    /// is computed from this rather than accumulated, so a dropped frame does
    /// not leave a drift behind.
    pub start: Transform,
}

/// How long a handle is, as a fraction of the distance from the camera.
///
/// The camera's vertical field of view is 60°, so the visible half-height at
/// distance *d* is about `0.58 d` — which makes this roughly a tenth of the
/// window's height, whatever the object is and wherever it is looked at from.
/// It was four times this to begin with, and the arrows ran off the screen.
const HANDLE: f32 = 0.12;

/// How close to a handle, in pixels, counts as grabbing it.
const TOLERANCE: f32 = 12.0;

pub struct ManipulatePlugin;

impl Plugin for ManipulatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GizmoDrag>()
            .add_systems(Update, (draw_handles, drag).chain());
    }
}

/// The three axis colours, matching the world axes the overlays draw.
const AXES: [(Vec3, Color); 3] = [
    (Vec3::X, Color::srgb(0.9, 0.3, 0.3)),
    (Vec3::Y, Color::srgb(0.3, 0.9, 0.3)),
    (Vec3::Z, Color::srgb(0.4, 0.5, 1.0)),
];

/// What the gizmo is attached to this frame, if anything.
///
/// `None` whenever there is no selection, no UI, or the selection is not an
/// object — an actor can be selected too, and an actor has no place of its own
/// to move.
fn target(
    selection: &Selection,
    objects: &Query<Selected, With<SceneObject>>,
    bounds: &Query<(&Aabb, &GlobalTransform)>,
) -> Option<Target> {
    let entity = selection.object?;
    let (id, world, local, children) = objects.get(entity).ok()?;
    Some(Target {
        object: id.0,
        origin: anchor(entity, children, world, bounds),
        local: *local,
    })
}

/// Where to put the handles.
///
/// **The middle of what is drawn, not the object's own origin.** An object is a
/// place in the tree and its translation is wherever a client put it, which for
/// anything loaded from a file is nowhere near the atoms — `gallery_demo.py`
/// offsets each structure by minus its own centre, so the origin can be a
/// hundred ångströms outside the thing it holds. Handles there look detached
/// from the object and are drawn through unrelated geometry.
///
/// The selection box already answers this question, so this asks it the same
/// way rather than a second way. Falling back to the origin covers an object
/// with nothing drawn under it, which has no bounds to take a middle of.
fn anchor(
    entity: Entity,
    children: Option<&Children>,
    world: &GlobalTransform,
    bounds: &Query<(&Aabb, &GlobalTransform)>,
) -> Vec3 {
    match super::overlays::subtree_bounds(entity, children, bounds) {
        // A box of no extent is a filter output that has never run — see
        // `viewport::has_extent`. Its "middle" is a point a fraction of a
        // millimetre from the world origin, which would drag the handles right
        // across the scene.
        Some((min, max)) if (max - min).max_element() > 0.0 => (min + max) * 0.5,
        _ => world.translation(),
    }
}

/// What the gizmo reads off the selected object.
type Selected = (
    &'static UniqueID,
    &'static GlobalTransform,
    &'static Transform,
    Option<&'static Children>,
);

/// The selected object, as the gizmo needs it.
///
/// The local transform is what a drag edits, because that is what
/// `SetTransform` writes and what the object's parent applies its own transform
/// to. `origin` is only where the handles are *drawn*.
struct Target {
    object: u64,
    origin: Vec3,
    local: Transform,
}

/// Draws the handles for the selected object.
///
/// Immediate mode, like every other overlay: there is nothing to keep between
/// frames, and a handle is a line whose length depends on where the camera is.
// Eight because the selection and the gizmo mode are two resources rather than
// one lump of interface state, which is the point of the split.
#[allow(clippy::too_many_arguments)]
fn draw_handles(
    mut gizmos: Gizmos,
    selection: Res<Selection>,
    gizmo: Res<GizmoMode>,
    held: Res<GizmoDrag>,
    objects: Query<Selected, With<SceneObject>>,
    bounds: Query<(&Aabb, &GlobalTransform)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    // Filtered to the 3D camera. An unfiltered `&Camera` query also matches
    // egui's 2D overlay camera, so `single` fails with "more than one" and the
    // whole system silently returns — which is exactly what it did, and why
    // nothing was ever drawn or grabbable. The axes visible before this were
    // `overlays`' read-only ones, drawn at the object's origin.
    camera: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
) {
    let mode = *gizmo;
    let Some(Target { origin, .. }) = target(&selection, &objects, &bounds) else {
        return;
    };
    let Ok((view_camera, view)) = camera.single() else {
        return;
    };
    let size = handle_size(view, origin);

    // Which handle the pointer is over, so it can be lit. Without this there is
    // no way to tell a grabbable handle from a decoration until you try — which
    // is exactly how it read the first time it was put in front of anyone.
    let hot = held.0.map(|grab| grab.axis).or_else(|| {
        let window = windows.single().ok()?;
        let ray = pointer_ray(window, view_camera, view)?;
        nearest_handle(origin, size, ray, window, view_camera, view).map(|(axis, _)| axis)
    });

    for (axis, colour) in AXES {
        // Lit rather than recoloured, so the axis is still identifiable.
        let colour = match hot == Some(axis) {
            true => Color::WHITE,
            false => colour,
        };
        match mode {
            GizmoMode::Translate => {
                gizmos.arrow(origin, origin + axis * size, colour);
            }
            GizmoMode::Scale => {
                let end = origin + axis * size;
                gizmos.line(origin, end, colour);
                gizmos.cube(
                    Transform::from_translation(end).with_scale(Vec3::splat(size * 0.12)),
                    colour,
                );
            }
            GizmoMode::Rotate => {
                gizmos.circle(
                    Isometry3d::new(origin, Quat::from_rotation_arc(Vec3::Z, axis)),
                    size,
                    colour,
                );
            }
        }
    }
}

/// How long a handle is, in world units.
///
/// Proportional to the distance from the camera, so it takes up the same amount
/// of *screen* whether the object is an atom across or a capsid across. A fixed
/// world size would be invisible on one and fill the view on the other.
fn handle_size(view: &GlobalTransform, origin: Vec3) -> f32 {
    view.translation().distance(origin) * HANDLE
}

/// The handle nearest the pointer, and how far away it is in pixels.
///
/// Shared by the hover highlight and the grab, so what lights up is exactly what
/// would be picked up. Two separate tests would eventually disagree, and the
/// disagreement would read as the gizmo ignoring clicks.
fn nearest_handle(
    origin: Vec3,
    size: f32,
    ray: Ray3d,
    window: &Window,
    camera: &Camera,
    view: &GlobalTransform,
) -> Option<(Vec3, f32)> {
    let pointer = pointer_in_view(window, camera)?;
    let mut best: Option<(Vec3, f32, f32)> = None;
    for (axis, _) in AXES {
        let Some(distance) = along_axis(origin, axis, ray) else {
            continue;
        };
        // Clamped to the drawn length: the axis is infinite and the handle is
        // not, so reaching past the arrowhead should miss.
        let on_axis = origin + axis * distance.clamp(0.0, size);
        let Ok(screen) = camera.world_to_viewport(view, on_axis) else {
            continue;
        };
        let away = screen.distance(pointer);
        if away < TOLERANCE && best.is_none_or(|(_, closest, _)| away < closest) {
            best = Some((axis, away, distance));
        }
    }
    best.map(|(axis, _, distance)| (axis, distance))
}

/// Where the pointer is, in the 3D camera's own viewport.
///
/// The camera is inset to whatever the panels leave, so a window position has
/// to be made relative to that rect before it means anything to
/// `viewport_to_world`.
fn pointer_ray(window: &Window, camera: &Camera, view: &GlobalTransform) -> Option<Ray3d> {
    camera
        .viewport_to_world(view, pointer_in_view(window, camera)?)
        .ok()
}

/// The pointer, if it is over the 3D camera's viewport at all.
///
/// **Returned in window coordinates, deliberately unadjusted.** The names are
/// misleading: despite being called `viewport_to_world` and `world_to_viewport`,
/// both of Bevy's conversions speak in *render-target* coordinates and apply the
/// viewport offset themselves — `world_to_viewport_core` ends
/// `… * target_rect.size() + target_rect.min`, and `viewport_to_ndc` opens
/// `(viewport_position - target_rect.min)`.
///
/// So subtracting `rect.min` here, as this did at first, takes the inset off
/// twice. The camera sits below the menu bar, so every handle tested one menu
/// bar too high and had to be approached from below. The rect is still needed —
/// but only to decide whether the pointer is over the 3D view or over a panel.
fn pointer_in_view(window: &Window, camera: &Camera) -> Option<Vec2> {
    let pointer = window.cursor_position()?;
    camera
        .logical_viewport_rect()?
        .contains(pointer)
        .then_some(pointer)
}

/// The point on an infinite line closest to a ray, as a distance along the line.
///
/// This is what makes a drag along an axis feel right: the object follows the
/// pointer's *projection* onto the axis rather than the pointer itself, so a
/// gesture across the screen moves it only as far along the axis as the gesture
/// actually went in that direction.
///
/// `None` when the ray and the axis are within a whisker of parallel, where the
/// closest point is unstable and a drag would fling the object.
fn along_axis(origin: Vec3, axis: Vec3, ray: Ray3d) -> Option<f32> {
    let direction = *ray.direction;
    // `w0` in the standard closest-approach derivation, pointing from the ray's
    // origin to the line's.
    let between = origin - ray.origin;
    let axis_dot_dir = axis.dot(direction);
    let determinant = 1.0 - axis_dot_dir * axis_dot_dir;
    if determinant.abs() < 1.0e-4 {
        return None;
    }
    // The signs matter and are easy to get backwards — this had them backwards
    // once, and the symptom was that nothing could be grabbed at all: every
    // point on a handle resolved to a negative distance, clamped to the origin,
    // and so sat nowhere near the pointer that was over it.
    //
    // For lines P = origin + t·axis and R = ray.origin + s·direction, the
    // parameter on P is (b·e − d) / (1 − b²), with b = axis·direction,
    // d = axis·w0 and e = direction·w0.
    Some((axis_dot_dir * direction.dot(between) - axis.dot(between)) / determinant)
}

/// Grabs, drags and releases.
#[allow(clippy::too_many_arguments)]
fn drag(
    buttons: Res<ButtonInput<MouseButton>>,
    mut held: ResMut<GizmoDrag>,
    captured: Res<super::PointerCaptured>,
    selection: Res<Selection>,
    gizmo: Res<GizmoMode>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    objects: Query<Selected, With<SceneObject>>,
    bounds: Query<(&Aabb, &GlobalTransform)>,
    bus: Res<CommandBus>,
    mut awake: ResMut<KeepAwake>,
) {
    if buttons.just_released(MouseButton::Left) {
        held.0 = None;
        return;
    }

    let Ok((view_camera, view)) = camera.single() else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(ray) = pointer_ray(window, view_camera, view) else {
        return;
    };

    // --- claiming a grab ------------------------------------------------
    if buttons.just_pressed(MouseButton::Left) && !captured.0 && held.0.is_none() {
        let Some(Target {
            object,
            origin,
            local,
        }) = target(&selection, &objects, &bounds)
        else {
            return;
        };
        let size = handle_size(view, origin);

        // The same test the highlight uses, so what lights up is what gets
        // picked up.
        if let Some((axis, offset)) = nearest_handle(origin, size, ray, window, view_camera, view) {
            held.0 = Some(Grab {
                object,
                axis,
                origin,
                offset,
                start: local,
            });
        }
        return;
    }

    // --- dragging -------------------------------------------------------
    let Some(grab) = held.0 else {
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        held.0 = None;
        return;
    }
    // Against the *pinned* origin, never the object's current one. See
    // `Grab::origin`.
    let Some(now) = along_axis(grab.origin, grab.axis, ray) else {
        return;
    };
    let moved = now - grab.offset;
    if moved.abs() < f32::EPSILON {
        return;
    }

    // Computed from the transform the drag started with, never accumulated: a
    // frame that does not run leaves no drift behind, and releasing and
    // grabbing again re-bases cleanly.
    let mode = *gizmo;
    let (translation, rotation, scale) = match mode {
        GizmoMode::Translate => (Some(grab.start.translation + grab.axis * moved), None, None),
        GizmoMode::Rotate => (
            None,
            Some(grab.start.rotation * Quat::from_axis_angle(grab.axis, moved * 0.05)),
            None,
        ),
        GizmoMode::Scale => {
            // Along the axis only, and never through zero — a scale of 0
            // flattens the object into a plane it cannot be dragged back out
            // of.
            let factor = (1.0 + moved * 0.05).max(0.01);
            let mut scaled = grab.start.scale;
            scaled += grab.axis * (grab.start.scale.dot(grab.axis) * (factor - 1.0));
            (None, None, Some(scaled))
        }
    };

    // The same command a script sends. Fire and forget: the reply says only
    // what was applied, and the next frame's `GlobalTransform` shows it.
    let (reply, _) = tokio::sync::oneshot::channel();
    let _ = bus.sender().send(SceneCommand::SetTransform {
        id: grab.object,
        translation,
        rotation,
        scale,
        reply,
    });

    // Reactive rendering: without this the drag advances only when something
    // else happens to wake the loop.
    awake.nudge();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sign is the whole test.
    ///
    /// It was backwards once and nothing caught it: the maths compiles either
    /// way, the handles still draw, and the only symptom is that a click beside
    /// a handle resolves to a point on the far side of the object and matches
    /// nothing. Invisible from outside, so it is pinned from inside.
    #[test]
    fn a_point_along_the_axis_comes_back_positive() {
        // The X axis through the origin, looked at from (5, 10, 0) downwards.
        // The nearest point on the axis is (5, 0, 0), which is +5 along it.
        let ray = Ray3d::new(Vec3::new(5.0, 10.0, 0.0), Dir3::NEG_Y);
        let along = along_axis(Vec3::ZERO, Vec3::X, ray).expect("not parallel");
        assert!((along - 5.0).abs() < 1.0e-4, "expected +5, got {along}");
    }

    /// And negative on the other side — which a formulation using a squared or
    /// absolute distance would fail while still passing the test above.
    #[test]
    fn the_other_side_of_the_origin_is_negative() {
        let ray = Ray3d::new(Vec3::new(-3.0, 10.0, 0.0), Dir3::NEG_Y);
        let along = along_axis(Vec3::ZERO, Vec3::X, ray).expect("not parallel");
        assert!((along + 3.0).abs() < 1.0e-4, "expected -3, got {along}");
    }

    /// Measured from the handle's own origin, not the world's. The handles sit
    /// at the middle of the selected object, which is rarely at (0, 0, 0).
    #[test]
    fn the_distance_is_measured_from_the_handles_own_origin() {
        let origin = Vec3::new(100.0, 0.0, 0.0);
        let ray = Ray3d::new(Vec3::new(104.0, 10.0, 0.0), Dir3::NEG_Y);
        let along = along_axis(origin, Vec3::X, ray).expect("not parallel");
        assert!((along - 4.0).abs() < 1.0e-4, "expected +4, got {along}");
    }

    /// Looking straight down an axis, every point on it is equally close and
    /// the answer is unstable. Refusing is what stops a drag flinging the
    /// object away when the camera happens to line up with a handle.
    #[test]
    fn a_ray_along_the_axis_is_refused() {
        let ray = Ray3d::new(Vec3::new(-10.0, 0.0, 0.0), Dir3::X);
        assert!(along_axis(Vec3::ZERO, Vec3::X, ray).is_none());
    }
}
