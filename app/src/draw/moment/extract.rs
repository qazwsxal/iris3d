//! Getting the volumes across to the render world.
//!
//! One flat list per frame rather than an entity per volume in the render
//! world. The moment pass draws every volume in one go with no sorting, no
//! batching and no per-entity state, so a list is all the pass can use — and
//! rebuilding it costs a matrix copy per volume, against a scene that is only
//! redrawn when something changes.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::render::Extract;

use super::MomentVolume;

/// One absorbing volume, ready to draw.
pub struct ExtractedVolume {
    pub mesh: AssetId<Mesh>,
    pub world_from_local: Mat4,
    pub sigma: f32,
    pub tint: Vec3,
    /// The eight world-space corners of the volume's bounds.
    ///
    /// Carried so the moment domain can be fitted to what is actually on
    /// screen. Corners rather than a centre and extent because the bound is
    /// wanted along the view axis, and a rotated box's extent along an
    /// arbitrary axis is not recoverable from its local one.
    pub corners: [Vec3; 8],
}

#[derive(Resource, Default)]
pub struct ExtractedVolumes(pub Vec<ExtractedVolume>);

/// Copies every visible absorbing volume into the render world.
///
/// Visibility is the inherited flag rather than the per-view one. The list is
/// shared by all views, so per-view frustum culling has nowhere to go here yet;
/// it belongs with the phase item this pass will eventually grow.
#[allow(clippy::type_complexity)]
pub fn extract_volumes(
    mut commands: Commands,
    volumes: Extract<
        Query<(
            &MomentVolume,
            &Mesh3d,
            &GlobalTransform,
            &InheritedVisibility,
            Option<&Aabb>,
        )>,
    >,
) {
    let extracted = volumes
        .iter()
        .filter(|(_, _, _, visibility, _)| visibility.get())
        .map(|(volume, mesh, transform, _, aabb)| {
            let world_from_local = transform.to_matrix();
            ExtractedVolume {
                mesh: mesh.0.id(),
                world_from_local,
                sigma: volume.sigma,
                tint: volume.tint,
                corners: corners_of(aabb, world_from_local),
            }
        })
        .collect();

    commands.insert_resource(ExtractedVolumes(extracted));
}

/// The world-space corners of a volume's local bounds.
///
/// A missing `Aabb` means the mesh has not been measured yet, which happens on
/// the frame it appears. A degenerate box at the origin of the volume is the
/// safe answer: it contributes nothing to the depth bound, and the next frame
/// has the real one.
fn corners_of(aabb: Option<&Aabb>, world_from_local: Mat4) -> [Vec3; 8] {
    let Some(aabb) = aabb else {
        return [world_from_local.transform_point3(Vec3::ZERO); 8];
    };
    let (low, high) = (Vec3::from(aabb.min()), Vec3::from(aabb.max()));
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let pick = |bit: usize, low: f32, high: f32| if index & (1 << bit) == 0 { low } else { high };
        *corner = world_from_local.transform_point3(Vec3::new(
            pick(0, low.x, high.x),
            pick(1, low.y, high.y),
            pick(2, low.z, high.z),
        ));
    }
    corners
}
