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
};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::view::{ViewTarget, ViewUniformOffset};

use super::extract::ExtractedVolumes;
use super::pipeline::{MomentResolvePipelineId, QueuedMomentPipelines};
use super::prepare::{MomentBindGroup, MomentResolveBindGroup, MomentTexture};

/// Accumulates every volume's signed optical depth into the moment target.
pub fn moment_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &MomentTexture,
        &MomentBindGroup,
        &ViewUniformOffset,
        Option<&MainPassResolutionOverride>,
    )>,
    volumes: Res<ExtractedVolumes>,
    queued: Res<QueuedMomentPipelines>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    mut ctx: RenderContext,
) {
    if volumes.0.is_empty() {
        return;
    }
    let (camera, moments, bind_group, view_uniform, resolution_override) = view.into_inner();

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("moment_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &moments.0.default_view,
            depth_slice: None,
            resolve_target: None,
            // Cleared to zero every frame: an empty pixel has absorbed nothing
            // and must reconstruct to `T = exp(0) = 1`.
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::NONE.into()),
                store: StoreOp::Store,
            },
        })],
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
        let Some(Some(pipeline_id)) = queued.0.get(index) else {
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
                pass.draw_indexed(
                    first..first + count,
                    vertices.range.start as i32,
                    instance,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertices.range.clone(), instance);
            }
        }
    }

    pass_span.end(&mut pass);
}

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
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &MomentResolveBindGroup,
        &MomentResolvePipelineId,
        Option<&MainPassResolutionOverride>,
    )>,
    volumes: Res<ExtractedVolumes>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if volumes.0.is_empty() {
        return;
    }
    let (camera, target, bind_group, pipeline_id, resolution_override) = view.into_inner();

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
    pass.set_bind_group(0, &bind_group.0, &[]);
    pass.draw(0..3, 0..1);

    pass_span.end(&mut pass);
}
