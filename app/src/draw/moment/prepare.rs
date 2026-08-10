//! Per-frame GPU resources: the moment target, the instance buffer, the bind
//! groups.

use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, Extent3d, PipelineCache, ShaderType, StorageBuffer,
    TextureDescriptor, TextureDimension, TextureUsages, UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniforms};

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

/// The accumulation targets for one view.
///
/// Two attachments, and the split between them is a budget decision rather
/// than a tidiness one. Four power moments of each colour channel would be
/// twelve floats plus totals — three `Rgba32Float` attachments at 48 bytes a
/// sample, over the 32-byte `max_color_attachment_bytes_per_sample` that many
/// adapters report. So:
///
/// - [`Self::moments`] holds `b1..b4` of a single *scalar* absorbance, which is
///   what the depth reconstruction runs on.
/// - [`Self::totals`] holds the per-channel total absorbance in RGB, and the
///   signed face count in alpha.
///
/// The assumption bought by that split is that a volume's *depth structure* is
/// the same in every channel while its *strength* differs. That is exactly true
/// for one tint and very close for several, and it keeps coloured volumes
/// composing correctly at 32 bytes a sample.
///
/// Alpha of `totals` earns its place as a correctness check rather than a
/// picture: a closed mesh contributes as many back faces as front faces, so any
/// pixel where it is not zero has an unclosed mesh or a dropped fragment.
///
/// Full `f32` rather than half. The accumulation is a difference of two similar
/// numbers, so a thin shell cancels catastrophically in `f16` — the reference
/// document's §8. The 16-bit quantisation tables in the MBOIT paper are for
/// Dirac-style surface fragments and do not apply to this formulation.
#[derive(Component)]
pub struct MomentTexture {
    pub moments: CachedTexture,
    pub totals: CachedTexture,
}

/// The interval of view-space depth the moment domain is fitted to.
///
/// Moments are only conditioned well over a bounded domain, so depth is warped
/// into `[0, 1]` across this interval before any power is taken. The warp is
/// linear, which is the cheapest choice that keeps the antiderivative in §3.3
/// closed form — a logarithmic warp would not, which §4 warns about directly.
///
/// Fitted per frame to the volumes actually on screen rather than left global.
/// A fixed bound spanning the whole scene would squeeze a small volume into a
/// sliver of the domain, and every moment past the first would then be a
/// difference of nearly equal numbers.
#[derive(Component, Clone, Copy, ShaderType)]
pub struct MomentBounds {
    pub near: f32,
    pub far: f32,
}

#[derive(Component)]
pub struct MomentBoundsBuffer(pub UniformBuffer<MomentBounds>);

#[allow(clippy::type_complexity)]
pub fn prepare_moment_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    volumes: Res<ExtractedVolumes>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &Msaa), With<MomentTransparency>>,
) {
    for (entity, camera, view, msaa) in &views {
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

        let mut target = |label| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some(label),
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
            )
        };

        let mut bounds = UniformBuffer::from(depth_bounds(&volumes, view));
        bounds.write_buffer(&render_device, &render_queue);

        commands.entity(entity).insert((
            MomentTexture {
                moments: target("moment_moments"),
                totals: target("moment_totals"),
            },
            MomentBoundsBuffer(bounds),
        ));
    }
}

/// Fits the moment domain to the volumes in front of this view.
///
/// Every volume's bounding corners are projected onto the view axis and the
/// extremes taken. Padded outwards by a little, because a warped depth of
/// exactly 0 or 1 sits on the edge of the domain the reconstruction is
/// conditioned over, and a fragment landing precisely there is the one most
/// likely to produce a degenerate Hankel matrix.
fn depth_bounds(volumes: &ExtractedVolumes, view: &ExtractedView) -> MomentBounds {
    let view_from_world = view.world_from_view.to_matrix().inverse();

    let mut near = f32::INFINITY;
    let mut far = f32::NEG_INFINITY;
    for volume in &volumes.0 {
        for corner in volume.corners {
            // Negated because the view looks down its own -Z, and the moment
            // domain wants a distance that grows away from the camera.
            let depth = -view_from_world.transform_point3(corner).z;
            near = near.min(depth);
            far = far.max(depth);
        }
    }

    // No volumes, or none measured yet. Any non-degenerate interval will do;
    // nothing will be accumulated into it.
    if !near.is_finite() || !far.is_finite() || far <= near {
        return MomentBounds {
            near: 0.0,
            far: 1.0,
        };
    }

    let padding = (far - near) * 0.05;
    MomentBounds {
        // Never behind the camera: depth is clamped to zero in the shader
        // anyway, and a negative near would waste half the domain on nothing.
        near: (near - padding).max(0.0),
        far: far + padding,
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
    views: Query<
        (
            Entity,
            &MomentTexture,
            &MomentBoundsBuffer,
            &ViewDepthTexture,
        ),
        With<MomentTransparency>,
    >,
) {
    let (Some(view_binding), Some(instance_binding)) =
        (view_uniforms.uniforms.binding(), instances.buffer.binding())
    else {
        return;
    };

    for (entity, moments, bounds, depth) in &views {
        let Some(bounds_binding) = bounds.0.binding() else {
            continue;
        };

        let accumulate = render_device.create_bind_group(
            "moment_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.moment_layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                instance_binding.clone(),
                depth.view(),
                bounds_binding.clone(),
            )),
        );
        let resolve = render_device.create_bind_group(
            "moment_resolve_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.resolve_layout),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &moments.moments.default_view,
                &moments.totals.default_view,
                depth.view(),
                bounds_binding,
            )),
        );
        commands
            .entity(entity)
            .insert((MomentBindGroup(accumulate), MomentResolveBindGroup(resolve)));
    }
}
