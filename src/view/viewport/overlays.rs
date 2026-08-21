//! Viewport overlays drawn with Bevy's immediate-mode gizmos.
//!
//! No extra dependency: `bevy_gizmos` ships with the engine, unlike
//! `transform-gizmo-bevy`, which is still pinned to Bevy 0.18. These are
//! read-only overlays — a selection outline, per-object axes, a ground grid —
//! rather than manipulation handles.
//!
//! Everything is redrawn every frame from the current world state, so there is
//! nothing to invalidate when an object moves, is rebuilt, or is deleted.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;

use crate::scene::SceneObject;
use crate::view::select::Selection;

/// Which overlays are switched on.
#[derive(Resource)]
pub struct OverlaySettings {
    pub grid: bool,
    pub world_axes: bool,
    pub selection: bool,
    /// Draw a faint outline around every object, not just the selected one.
    pub all_bounds: bool,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        Self {
            grid: true,
            world_axes: true,
            selection: true,
            all_bounds: false,
        }
    }
}

const SELECTED: Color = Color::srgb(1.0, 0.72, 0.2);
const UNSELECTED: Color = Color::srgb(0.35, 0.38, 0.45);
const GRID: Color = Color::srgb(0.24, 0.25, 0.29);

pub fn draw_overlays(
    mut gizmos: Gizmos,
    settings: Res<OverlaySettings>,
    // Registered by `ViewportPlugin`, not by the interface, so the grid and
    // axes still draw in a build with no `UiPlugin`. Nothing is selected when
    // there is no interface to select in, which is the right answer rather than
    // a missing resource.
    selection: Res<Selection>,
    objects: Query<(Entity, &GlobalTransform, Option<&Children>), With<SceneObject>>,
    bounds: Query<(&Aabb, &GlobalTransform)>,
) {
    if settings.grid {
        // Sized to the scene so it stays useful whether the data is angstroms
        // or metres, rather than a fixed grid that is invisible or enormous.
        let extent = scene_extent(&bounds).unwrap_or(10.0);
        let spacing = nice_step(extent / 10.0);
        let cells = ((extent / spacing).ceil() as u32).clamp(4, 200) * 2;
        gizmos
            .grid(
                Isometry3d::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                UVec2::splat(cells),
                Vec2::splat(spacing),
                GRID,
            )
            .outer_edges();
    }

    if settings.world_axes {
        let length = scene_extent(&bounds).unwrap_or(10.0) * 0.15;
        gizmos.line(Vec3::ZERO, Vec3::X * length, Color::srgb(0.9, 0.3, 0.3));
        gizmos.line(Vec3::ZERO, Vec3::Y * length, Color::srgb(0.3, 0.9, 0.3));
        gizmos.line(Vec3::ZERO, Vec3::Z * length, Color::srgb(0.35, 0.5, 1.0));
    }

    for (entity, transform, children) in &objects {
        let selected = selection.object == Some(entity);
        if !selected && !settings.all_bounds {
            continue;
        }
        if selected && !settings.selection {
            continue;
        }

        // An object entity carries no mesh of its own — the geometry belongs
        // to its actor children, so the outline is their combined extent.
        let Some((min, max)) = subtree_bounds(entity, children, &bounds) else {
            continue;
        };
        let centre = (min + max) * 0.5;
        let size = max - min;
        // Defensive: a box with no extent renders as a speck rather than
        // anything useful.
        if size.max_element() < 1e-3 {
            continue;
        }
        let colour = if selected { SELECTED } else { UNSELECTED };

        gizmos.primitive_3d(
            &Cuboid::from_size(size),
            Isometry3d::from_translation(centre),
            colour,
        );

        if selected {
            // Axes at the object's own origin, which is not generally the
            // centre of its bounds.
            gizmos.axes(*transform, size.length() * 0.25);
        }
    }
}

/// Union of the Aabbs on an object and whatever is parented to it.
///
/// Walks `Children`, not `Actors`: this is a question about space, and what
/// belongs in an object's box is what is *drawn at* its placement. A actor
/// sourced from elsewhere and parented here counts; one sourced from here but
/// parented elsewhere does not, because that is where it appears.
pub(super) fn subtree_bounds(
    entity: Entity,
    children: Option<&Children>,
    bounds: &Query<(&Aabb, &GlobalTransform)>,
) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut found = false;

    let consider = |entity: Entity, min: &mut Vec3, max: &mut Vec3, found: &mut bool| {
        if let Ok((aabb, global)) = bounds.get(entity) {
            let centre = global.transform_point(Vec3::from(aabb.center));
            let extent = (global.affine().matrix3 * Vec3::from(aabb.half_extents)).abs();
            *min = min.min(centre - extent);
            *max = max.max(centre + extent);
            *found = true;
        }
    };

    consider(entity, &mut min, &mut max, &mut found);
    for child in children.into_iter().flatten() {
        consider(*child, &mut min, &mut max, &mut found);
    }

    found.then_some((min, max))
}

/// Rough size of everything on screen, used to scale the grid and axes.
///
/// Skips boxes with no extent for the reason
/// [`has_extent`](super::has_extent) gives: a filter's geometry output is a
/// placeholder until it runs, and one that never runs stays a point at the
/// origin. Sizing the grid off that collapses its spacing to nothing, so the
/// grid disappears at the same moment the camera does — two symptoms, one
/// cause, and neither of them says which.
fn scene_extent(bounds: &Query<(&Aabb, &GlobalTransform)>) -> Option<f32> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for (aabb, global) in bounds.iter().filter(|(aabb, _)| super::has_extent(aabb)) {
        let centre = global.transform_point(Vec3::from(aabb.center));
        let extent = (global.affine().matrix3 * Vec3::from(aabb.half_extents)).abs();
        min = min.min(centre - extent);
        max = max.max(centre + extent);
    }
    (min.is_finite() && max.is_finite()).then(|| (max - min).length().max(0.001))
}

/// Rounds to a 1/2/5 x 10^n step, so grid lines land on readable values.
fn nice_step(raw: f32) -> f32 {
    if raw <= 0.0 || !raw.is_finite() {
        return 1.0;
    }
    let magnitude = 10f32.powf(raw.log10().floor());
    let normalised = raw / magnitude;
    let step = if normalised < 1.5 {
        1.0
    } else if normalised < 3.5 {
        2.0
    } else if normalised < 7.5 {
        5.0
    } else {
        10.0
    };
    step * magnitude
}

#[cfg(test)]
mod tests {
    use super::nice_step;

    #[test]
    fn steps_round_to_readable_values() {
        assert_eq!(nice_step(0.9), 1.0);
        assert_eq!(nice_step(1.2), 1.0);
        assert_eq!(nice_step(2.0), 2.0);
        assert_eq!(nice_step(4.0), 5.0);
        assert_eq!(nice_step(8.0), 10.0);
        assert_eq!(nice_step(23.0), 20.0);
        // 3 is not a 1/2/5 step, so it rounds down to 2.
        assert!((nice_step(0.03) - 0.02).abs() < 1e-6);
    }

    #[test]
    fn steps_survive_degenerate_input() {
        assert_eq!(nice_step(0.0), 1.0);
        assert_eq!(nice_step(-5.0), 1.0);
        assert_eq!(nice_step(f32::NAN), 1.0);
    }
}
