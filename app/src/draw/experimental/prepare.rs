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

use super::MomentView;
use super::extract::{ExtractedShellLighting, ExtractedVolumes, ShellLighting};
use super::pipeline::{MOMENT_FORMAT, MomentPipelines};

/// What one volume needs in the vertex and fragment stages.
///
/// Indexed by `@builtin(instance_index)`, which the pass sets per draw. That
/// costs nothing and saves a dynamic-offset rebind between volumes.
/// One buffer for both passes rather than one each. The accumulation ignores
/// the last three fields and the shell ignores `sigma`, which costs a few bytes
/// per volume against keeping two buffers in the same order.
#[derive(Clone, Copy, ShaderType)]
pub struct MomentInstance {
    pub world_from_local: Mat4,
    pub world_from_local_normal: Mat3,
    pub tint: Vec3,
    /// Extinction per unit for an interior, or the spike's absorbance for a
    /// film. Which one is decided by `dirac`.
    pub strength: f32,
    /// Zero for an interior, one for a film. See
    /// [`Depiction`](super::Depiction).
    pub dirac: u32,
    pub f0: f32,
    pub roughness: f32,
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
/// - [`Self::totals`] holds the per-channel total absorbance in RGB, and `b0`
///   of the scalar measure in alpha.
///
/// The assumption bought by that split is that a volume's *depth structure* is
/// the same in every channel while its *strength* differs. That is exactly true
/// for one tint and very close for several, and it keeps coloured volumes
/// composing correctly at 32 bytes a sample.
///
/// Full `f32` rather than half. The accumulation is a difference of two similar
/// numbers, so a thin shell cancels catastrophically in `f16` — the reference
/// document's §8. The 16-bit quantisation tables in the MBOIT paper are for
/// Dirac-style surface fragments and do not apply to this formulation.
///
/// Both take the view's sample count, so a multisampled view accumulates a
/// separate absorbance per sample and the resolve can dim each sample of the
/// target by its own transmittance. Without that the volume's silhouette would
/// be the one hard edge in an otherwise antialiased picture.
#[derive(Component)]
pub struct MomentTexture {
    pub moments: CachedTexture,
    pub totals: CachedTexture,
    /// The view's sample count, carried so the pass and the bind groups pick
    /// the pipeline and layout that match rather than asking `Msaa` again.
    pub samples: u32,
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
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &Msaa), With<MomentView>>,
) {
    for (entity, camera, view, msaa) in &views {
        let Some(size) = camera.physical_target_size else {
            continue;
        };
        let samples = msaa.samples();

        // Matched to the view. The accumulation writes here through a pipeline
        // of the same sample count, and the resolve reads it per sample.
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
                    sample_count: samples,
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
                samples,
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
        world_from_local_normal: volume.world_from_local_normal,
        tint: volume.tint,
        strength: volume.strength,
        dirac: volume.dirac,
        f0: volume.shell.map_or(0.0, |shell| shell.f0),
        roughness: volume.shell.map_or(1.0, |shell| shell.roughness),
    }));

    // An empty storage buffer is not a valid binding, so keep one dead entry
    // rather than teaching every consumer to cope with a missing bind group.
    if values.is_empty() {
        values.push(MomentInstance {
            world_from_local: Mat4::IDENTITY,
            world_from_local_normal: Mat3::IDENTITY,
            tint: Vec3::ZERO,
            strength: 0.0,
            dirac: 0,
            f0: 0.0,
            roughness: 1.0,
        });
    }

    instances.buffer.write_buffer(&render_device, &render_queue);
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

/// Bind group 0 of the shell pass: the view, the instances, and the lights.
///
/// No textures, so unlike the other two this one does not change shape with the
/// sample count — the shell is an ordinary surface pass that happens to blend
/// additively, and it reads neither the depth buffer nor the moments.
#[derive(Component)]
pub struct ShellBindGroup(pub BindGroup);

/// The lights, uploaded once per frame and shared by every view.
#[derive(Resource, Default)]
pub struct ShellLightingBuffer(pub UniformBuffer<ShellLighting>);

pub fn prepare_shell_lighting(
    mut buffer: ResMut<ShellLightingBuffer>,
    lighting: Res<ExtractedShellLighting>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffer.0.set(lighting.0.clone());
    buffer.0.write_buffer(&render_device, &render_queue);
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_shell_bind_groups(
    mut commands: Commands,
    pipelines: Res<MomentPipelines>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    view_uniforms: Res<ViewUniforms>,
    instances: Res<MomentInstances>,
    lighting: Res<ShellLightingBuffer>,
    views: Query<(Entity, &MomentTexture, &MomentBoundsBuffer), With<MomentView>>,
) {
    let (Some(view_binding), Some(instance_binding), Some(lighting_binding)) = (
        view_uniforms.uniforms.binding(),
        instances.buffer.binding(),
        lighting.0.binding(),
    ) else {
        return;
    };

    for (entity, moments, bounds) in &views {
        let Some(bounds_binding) = bounds.0.binding() else {
            continue;
        };
        let bind_group = render_device.create_bind_group(
            "moment_shell_bind_group",
            &pipeline_cache.get_bind_group_layout(pipelines.shell_layout(moments.samples)),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                instance_binding.clone(),
                lighting_binding.clone(),
                &moments.moments.default_view,
                &moments.totals.default_view,
                bounds_binding,
            )),
        );
        commands.entity(entity).insert(ShellBindGroup(bind_group));
    }
}

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
        With<MomentView>,
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
            &pipeline_cache.get_bind_group_layout(pipelines.moment_layout(moments.samples)),
            &BindGroupEntries::sequential((
                view_binding.clone(),
                instance_binding.clone(),
                depth.view(),
                bounds_binding.clone(),
            )),
        );
        let resolve = render_device.create_bind_group(
            "moment_resolve_bind_group",
            &pipeline_cache.get_bind_group_layout(pipelines.resolve_layout(moments.samples)),
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
