//! Per-frame GPU resources: the moment target, the instance buffer, the bind
//! groups.

use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, Extent3d, PipelineCache, ShaderType, StorageBuffer,
    TextureDescriptor, TextureDimension, TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{Msaa, ViewDepthTexture, ViewUniforms};

use super::extract::ExtractedVolumes;
use super::pipeline::{MomentPipelines, MOMENT_FORMAT};
use super::MomentTransparency;

/// What one volume needs in the vertex and fragment stages.
///
/// Indexed by `@builtin(instance_index)`, which the pass sets per draw. That
/// costs nothing and saves a dynamic-offset rebind between volumes.
#[derive(Clone, Copy, ShaderType)]
pub struct MomentInstance {
    pub world_from_local: Mat4,
    pub tint: Vec3,
    pub sigma: f32,
}

#[derive(Resource, Default)]
pub struct MomentInstances {
    pub buffer: StorageBuffer<Vec<MomentInstance>>,
}

/// The accumulation target for one view.
///
/// RGB carry the signed optical depth of each colour channel, which is what
/// makes two differently tinted volumes compose correctly rather than sharing
/// one global colour. Alpha carries the signed *face count*, and earns its
/// place as a correctness check rather than a picture: a closed mesh
/// contributes exactly as many back faces as front faces, so any pixel where
/// alpha is not zero has an unclosed mesh or a dropped fragment behind it.
///
/// Full `f32` rather than half. The accumulation is a difference of two similar
/// numbers, so a thin shell cancels catastrophically in `f16` — the reference
/// document's §8. The 16-bit quantisation tables in the MBOIT paper are for
/// Dirac-style surface fragments and do not apply to this formulation.
#[derive(Component)]
pub struct MomentTexture(pub CachedTexture);

pub fn prepare_moment_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &Msaa), With<MomentTransparency>>,
) {
    for (entity, camera, msaa) in &views {
        let Some(size) = camera.physical_target_size else {
            continue;
        };
        // A multisampled view would need a multisampled moment target and a
        // resolve that reads per sample. That is real work rather than a
        // missing line, so the view is skipped and warned about in the main
        // world instead of half-supported here.
        if msaa.samples() > 1 {
            continue;
        }

        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("moment_accumulation"),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MOMENT_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(MomentTexture(texture));
    }
}

pub fn prepare_moment_instances(
    mut instances: ResMut<MomentInstances>,
    volumes: Res<ExtractedVolumes>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let values = instances.buffer.get_mut();
    values.clear();
    values.extend(volumes.0.iter().map(|volume| MomentInstance {
        world_from_local: volume.world_from_local,
        tint: volume.tint,
        sigma: volume.sigma,
    }));

    // An empty storage buffer is not a valid binding, so keep one dead entry
    // rather than teaching every consumer to cope with a missing bind group.
    if values.is_empty() {
        values.push(MomentInstance {
            world_from_local: Mat4::IDENTITY,
            tint: Vec3::ZERO,
            sigma: 0.0,
        });
    }

    instances
        .buffer
        .write_buffer(&render_device, &render_queue);
}

/// Bind group 0 of the accumulation pass: the view, every volume's instance
/// data, and the opaque depth buffer.
///
/// Per view rather than a resource because of that last one. The pass clamps
/// each endpoint against opaque depth in the shader instead of depth-testing
/// (see [`super::pass`]), so the depth texture is an ordinary binding, and
/// there is one of those per view.
#[derive(Component)]
pub struct MomentBindGroup(pub BindGroup);

/// Bind group 0 of the resolve pass. Per view for the same reason.
#[derive(Component)]
pub struct MomentResolveBindGroup(pub BindGroup);

pub fn prepare_moment_bind_groups(
    mut commands: Commands,
    pipelines: Res<MomentPipelines>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    instances: Res<MomentInstances>,
    views: Query<(Entity, &MomentTexture, &ViewDepthTexture), With<MomentTransparency>>,
) {
    let (Some(view_binding), Some(instance_binding)) =
        (view_uniforms.uniforms.binding(), instances.buffer.binding())
    else {
        return;
    };

    for (entity, moments, depth) in &views {
        let accumulate = render_device.create_bind_group(
            "moment_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.moment_layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                instance_binding.clone(),
                depth.view(),
            )),
        );
        let resolve = render_device.create_bind_group(
            "moment_resolve_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.resolve_layout),
            &BindGroupEntries::single(&moments.0.default_view),
        );
        commands
            .entity(entity)
            .insert((MomentBindGroup(accumulate), MomentResolveBindGroup(resolve)));
    }
}
