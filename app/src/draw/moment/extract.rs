//! Getting the volumes across to the render world.
//!
//! One flat list per frame rather than an entity per volume in the render
//! world. The moment pass draws every volume in one go with no sorting, no
//! batching and no per-entity state, so a list is all the pass can use — and
//! rebuilding it costs a matrix copy per volume, against a scene that is only
//! redrawn when something changes.

use bevy::prelude::*;
use bevy::render::Extract;

use super::MomentVolume;

/// One absorbing volume, ready to draw.
pub struct ExtractedVolume {
    pub mesh: AssetId<Mesh>,
    pub world_from_local: Mat4,
    pub sigma: f32,
    pub tint: Vec3,
}

#[derive(Resource, Default)]
pub struct ExtractedVolumes(pub Vec<ExtractedVolume>);

/// Copies every visible absorbing volume into the render world.
///
/// Visibility is the inherited flag rather than the per-view one. The list is
/// shared by all views, so per-view frustum culling has nowhere to go here yet;
/// it belongs with the phase item this pass will eventually grow.
pub fn extract_volumes(
    mut commands: Commands,
    volumes: Extract<Query<(&MomentVolume, &Mesh3d, &GlobalTransform, &InheritedVisibility)>>,
) {
    let extracted = volumes
        .iter()
        .filter(|(_, _, _, visibility)| visibility.get())
        .map(|(volume, mesh, transform, _)| ExtractedVolume {
            mesh: mesh.0.id(),
            world_from_local: transform.to_matrix(),
            sigma: volume.sigma,
            tint: volume.tint,
        })
        .collect();

    commands.insert_resource(ExtractedVolumes(extracted));
}
