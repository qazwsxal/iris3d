//! The two passes.
//!
//! # Why there is no depth attachment
//!
//! The reference document's §9 says to depth-test the accumulation pass against
//! the opaque buffer with `depth_write_enabled: false`. That is wrong for this
//! formulation, and wrong twice over.
//!
//! First the direction: Bevy uses a reverse-Z projection, so the comparison
//! would have to be `GreaterEqual`, not `Less`. Second, and worse, depth
//! testing is the wrong *mechanism*. A fragment here is not a sample of colour
//! but one endpoint of an interval. Discard a back face because a wall stands
//! in front of it and its front face still contributes `-F(z_in)`: the interval
//! never closes and the pixel accumulates a negative thickness, which shows up
//! as a bright halo around anything intersecting opaque geometry.
//!
//! §3.3 has the right rule. Clamp *both* endpoints to the opaque depth, so an
//! interval entirely behind a wall collapses to `F(z) - F(z) = 0` and one
//! crossing the wall is truncated at it. Additivity survives, and no fragment
//! is thrown away. So the pass binds the depth buffer as a texture, samples it,
//! and clamps — see `moment.wgsl`.

use bevy::camera::{MainPassResolutionOverride, Viewport};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    LoadOp, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
    TextureView,
};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget, ViewUniformOffset};

use super::extract::{ExtractedGrids, ExtractedVolumes};
use super::pipeline::{
    MomentResolvePipelineId, QueuedGridPipelines, QueuedMomentPipelines, QueuedShellPipelines,
    ShellKey,
};
use super::prepare::{
    GridBindGroups, MomentBindGroup, MomentResolveBindGroup, MomentTexture, ShellBindGroup,
};

/// One of the accumulation attachments, cleared and kept.
fn attachment(view: &TextureView) -> RenderPassColorAttachment<'_> {
    RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: Operations {
            load: LoadOp::Clear(LinearRgba::NONE.into()),
            store: StoreOp::Store,
        },
    }
}

/// What the accumulation needs of the view it is drawing for.
///
/// The entity comes first so the per-view grid pipeline can be looked up; the
/// mesh half needs only the sample count, which [`MomentTexture`] carries.
type AccumulateView = (
    Entity,
    &'static ExtractedCamera,
    &'static MomentTexture,
    &'static MomentBindGroup,
    &'static ViewUniformOffset,
    Option<&'static MainPassResolutionOverride>,
);

/// Accumulates every volume's signed optical depth into the moment target.
#[allow(clippy::too_many_arguments)]
pub fn moment_pass(
    view: ViewQuery<AccumulateView>,
    volumes: Res<ExtractedVolumes>,
    grids: Res<ExtractedGrids>,
    grid_bindings: Res<GridBindGroups>,
    queued: Res<QueuedMomentPipelines>,
    queued_grids: Res<QueuedGridPipelines>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    mut ctx: RenderContext,
) {
    // Both kinds accumulate into the same target, so either one alone is reason
    // enough to run. Testing only the meshes left a scene of pure volumes
    // clearing no target and resolving against stale contents.
    if volumes.0.is_empty() && grids.0.is_empty() {
        return;
    }
    let (view_entity, camera, moments, bind_group, view_uniform, resolution_override) =
        view.into_inner();

    // The list built for this view's sample count: a pipeline written for a
    // single-sample target cannot draw into a multisampled one. Looked up
    // before the pass is begun, so an early return here does not leave a
    // diagnostic span open.
    let Some(queued) = queued.0.get(&moments.samples) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("moment_pass"),
        // Cleared to zero every frame: an empty pixel has absorbed nothing and
        // must reconstruct to `T = exp(0) = 1`.
        color_attachments: &[
            Some(attachment(&moments.moments.default_view)),
            Some(attachment(&moments.totals.default_view)),
        ],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut pass, "moment_pass");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }
    pass.set_bind_group(0, &bind_group.0, &[view_uniform.offset]);

    for (index, volume) in volumes.0.iter().enumerate() {
        let Some(Some(pipeline_id)) = queued.get(index) else {
            continue;
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(*pipeline_id) else {
            continue;
        };
        let Some(mesh) = meshes.get(volume.mesh) else {
            continue;
        };
        let Some(vertices) = mesh_allocator.mesh_vertex_slice(&volume.mesh) else {
            continue;
        };

        pass.set_render_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));

        // One instance per draw, so `@builtin(instance_index)` in the shader is
        // this volume's index into the instance buffer. Free, and it saves a
        // dynamic-offset rebind between volumes.
        let instance = index as u32..index as u32 + 1;

        match &mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(indices) = mesh_allocator.mesh_index_slice(&volume.mesh) else {
                    continue;
                };
                pass.set_index_buffer(indices.buffer.slice(..), *index_format);
                let first = indices.range.start;
                pass.draw_indexed(first..first + count, vertices.range.start as i32, instance);
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertices.range.clone(), instance);
            }
        }
    }

    // Grids, into the same target and under the same group 0. Each is its own
    // draw because each has its own field texture, and each is a fullscreen
    // triangle rather than geometry — see the header of `volume.wgsl`.
    if let Some((accumulate, _)) = queued_grids.0.get(&view_entity)
        && let Some(pipeline) = pipeline_cache.get_render_pipeline(*accumulate)
    {
        pass.set_render_pipeline(pipeline);
        for binding in &grid_bindings.0 {
            pass.set_bind_group(1, &binding.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pass_span.end(&mut pass);
}

/// What the shell needs of the view it is drawing into.
type ShellView = (
    &'static ExtractedCamera,
    &'static ViewTarget,
    &'static ViewDepthTexture,
    &'static ShellBindGroup,
    &'static ExtractedView,
    &'static Msaa,
    &'static ViewUniformOffset,
    Option<&'static MainPassResolutionOverride>,
);

/// Draws the dielectric shell over whatever the resolve left behind.
///
/// After the resolve, and that ordering is the physical claim: light reflecting
/// off the near face of a glass object never enters the medium, so it must not
/// be dimmed by the medium's own absorbance. Running before the resolve would
/// multiply the highlight by the transmittance of the very volume it is
/// bouncing off.
///
/// What the shell *is* dimmed by is whatever medium lies in front of it, which
/// it works out per fragment from the moments — see `transmittance_in_front` in
/// `shell.wgsl`. That is the general rule this pathway composites by, and the
/// fullscreen resolve is the special case of it: the accumulation clamps every
/// interval to the opaque depth, so for an opaque surface the fraction in front
/// is exactly one and the total is already the answer. Anything at an
/// *intermediate* depth has to ask the moments instead, and the shell is the
/// first thing here that does.
pub fn shell_pass(
    view: ViewQuery<ShellView>,
    volumes: Res<ExtractedVolumes>,
    queued: Res<QueuedShellPipelines>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    mut ctx: RenderContext,
) {
    if volumes.0.is_empty() {
        return;
    }
    let (camera, target, depth, bind_group, extracted, msaa, view_uniform, resolution_override) =
        view.into_inner();

    let key = ShellKey {
        samples: msaa.samples(),
        format: extracted.target_format,
    };
    let Some(queued) = queued.0.get(&key) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("moment_shell"),
        color_attachments: &[Some(target.get_color_attachment())],
        // Read-only: the shell tests against opaque depth but writes none, so
        // it neither occludes the volumes nor anything drawn after it.
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut pass, "moment_shell");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }
    pass.set_bind_group(0, &bind_group.0, &[view_uniform.offset]);

    for (index, volume) in volumes.0.iter().enumerate() {
        let Some(Some(pipeline_id)) = queued.get(index) else {
            continue;
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(*pipeline_id) else {
            continue;
        };
        let Some(mesh) = meshes.get(volume.mesh) else {
            continue;
        };
        let Some(vertices) = mesh_allocator.mesh_vertex_slice(&volume.mesh) else {
            continue;
        };

        pass.set_render_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        let instance = index as u32..index as u32 + 1;

        match &mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(indices) = mesh_allocator.mesh_index_slice(&volume.mesh) else {
                    continue;
                };
                pass.set_index_buffer(indices.buffer.slice(..), *index_format);
                let first = indices.range.start;
                pass.draw_indexed(first..first + count, vertices.range.start as i32, instance);
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertices.range.clone(), instance);
            }
        }
    }

    pass_span.end(&mut pass);
}

/// What the resolve needs of the view it is dimming.
type ResolveView = (
    &'static ExtractedCamera,
    &'static ViewTarget,
    &'static MomentResolveBindGroup,
    &'static MomentResolvePipelineId,
    &'static ViewUniformOffset,
    Option<&'static MainPassResolutionOverride>,
);

/// Reconstructs transmittance and dims the view target by it.
///
/// A fullscreen triangle rather than a second pass over the geometry. At
/// `k = 0` the accumulated value is the *whole* optical depth along the ray, so
/// there is nothing left to evaluate per fragment — the moment target already
/// holds the answer for every pixel. That stops being true as soon as higher
/// moments arrive: reconstruction is then a function of depth, each fragment
/// needs its own `T(z)`, and this becomes the second geometry pass the
/// reference document's §2 describes.
pub fn moment_resolve(
    view: ViewQuery<ResolveView>,
    volumes: Res<ExtractedVolumes>,
    grids: Res<ExtractedGrids>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    // Either kind deposited absorbance worth applying. Guarding on the meshes
    // alone would leave a scene of pure volumes accumulated but never resolved.
    if volumes.0.is_empty() && grids.0.is_empty() {
        return;
    }
    let (camera, target, bind_group, pipeline_id, view_uniform, resolution_override) =
        view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("moment_resolve"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut pass, "moment_resolve");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group.0, &[view_uniform.offset]);
    pass.draw(0..3, 0..1);

    pass_span.end(&mut pass);
}

/// What the emission pass needs of the view it is drawing into.
///
/// The resolve's bindings exactly, because `emit.wgsl` reuses that layout — see
/// the note on group 0 in the shader. The entity comes along so the per-view
/// pipeline can be looked up.
type EmitView = (
    Entity,
    &'static ExtractedCamera,
    &'static ViewTarget,
    &'static MomentResolveBindGroup,
    &'static ViewUniformOffset,
    Option<&'static MainPassResolutionOverride>,
);

/// Adds what each volume emits, attenuated by whatever stands in front of it.
///
/// After the resolve, and for the same physical reason the shell is: the resolve
/// has already dimmed everything *behind* the volume by `exp(-A)`, and this is
/// the light the volume puts back on top. Running it before the resolve would
/// multiply the volume's own emission by its own absorbance a second time —
/// `emit.wgsl` already accounts for that internally, marching front to back.
///
/// Beside the shell rather than before or after it. Both are additive, so their
/// order between themselves does not matter; what matters is that both come
/// after the resolve.
///
/// One draw per grid, each a fullscreen triangle. There is no depth attachment
/// here either: the emission integral is *truncated* at an occluder rather than
/// rejected by it, which the shader does by clamping against the depth texture.
pub fn grid_emit_pass(
    view: ViewQuery<EmitView>,
    grids: Res<ExtractedGrids>,
    grid_bindings: Res<GridBindGroups>,
    queued_grids: Res<QueuedGridPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if grids.0.is_empty() || grid_bindings.0.is_empty() {
        return;
    }
    let (view_entity, camera, target, bind_group, view_uniform, resolution_override) =
        view.into_inner();

    let Some((_, emit)) = queued_grids.0.get(&view_entity) else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_render_pipeline(*emit) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("moment_grid_emit"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass_span = diagnostics.pass_span(&mut pass, "moment_grid_emit");

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        pass.set_camera_viewport(&viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group.0, &[view_uniform.offset]);
    for binding in &grid_bindings.0 {
        pass.set_bind_group(1, &binding.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pass_span.end(&mut pass);
}
