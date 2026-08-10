//! The two pipelines: accumulate, then reconstruct.

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::entity::EntityHashMap;
use bevy::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, texture_2d, texture_depth_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation,
    BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, PipelineCache,
    PrimitiveState, RenderPipelineDescriptor, ShaderStages, SpecializedMeshPipeline,
    SpecializedMeshPipelineError, SpecializedMeshPipelines, TextureFormat, TextureSampleType,
    VertexState,
};
use bevy::render::view::{ExtractedView, ViewUniform};
use bevy::shader::Shader;

use super::extract::ExtractedVolumes;
use super::MomentTransparency;

/// The moment target's format, fixed here so the pipeline and the texture
/// cannot disagree.
pub const MOMENT_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

#[derive(Resource)]
pub struct MomentPipelines {
    pub moment_layout: BindGroupLayoutDescriptor,
    pub resolve_layout: BindGroupLayoutDescriptor,
    pub resolve_shader: Handle<Shader>,
    pub fullscreen: FullscreenShader,
}

pub fn init_moment_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen: Res<FullscreenShader>,
) {
    let moment_layout = BindGroupLayoutDescriptor::new(
        "moment_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                storage_buffer_read_only_sized(false, None),
                texture_depth_2d(),
            ),
        ),
    );

    let resolve_layout = BindGroupLayoutDescriptor::new(
        "moment_resolve_bind_group_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            // Never filtered, and loaded by integer coordinate. Interpolating
            // moments between neighbouring pixels is only meaningful in a
            // light-space map, where prefiltering is the whole advantage; in
            // view space it mixes unrelated depth distributions.
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );

    let moment_shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "moment.wgsl");

    commands.insert_resource(MomentMeshPipeline {
        layout: moment_layout.clone(),
        shader: moment_shader.clone(),
    });
    commands.insert_resource(MomentPipelines {
        moment_layout,
        resolve_layout,
        resolve_shader: load_embedded_asset!(asset_server.as_ref(), "resolve.wgsl"),
        fullscreen: fullscreen.clone(),
    });
}

/// Specialises the accumulation pipeline over the mesh's vertex layout.
///
/// Only the position is pulled. The moment pass has no lighting, no colour and
/// no texturing, so normals and vertex colours are dead weight in it — but the
/// meshes come from other backends and carry them, so the stride still has to
/// be read off the real layout rather than assumed.
#[derive(Resource)]
pub struct MomentMeshPipeline {
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

impl SpecializedMeshPipeline for MomentMeshPipeline {
    type Key = ();

    fn specialize(
        &self,
        _key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_layout =
            layout.0.get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;

        Ok(RenderPipelineDescriptor {
            label: Some("moment_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format: MOMENT_FORMAT,
                    // Additive. This is the one line that makes the whole
                    // method order-independent.
                    blend: Some(BlendState {
                        color: ADDITIVE,
                        alpha: ADDITIVE,
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            // Both signs in one draw: the fragment shader branches on
            // `front_facing` and negates.
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            // No depth attachment, deliberately. The reference document's §9
            // says to depth-test against the opaque buffer, but that is wrong
            // for this formulation: discarding a back face while keeping its
            // front face leaves the interval unclosed and the pixel
            // accumulates a negative thickness. §3.3 has the right rule —
            // *clamp* both endpoints to the opaque depth — and the fragment
            // shader does that by sampling the depth texture instead.
            depth_stencil: None,
            ..default()
        })
    }
}

const ADDITIVE: BlendComponent = BlendComponent {
    src_factor: BlendFactor::One,
    dst_factor: BlendFactor::One,
    operation: BlendOperation::Add,
};

/// `dst = 0 * src + src * dst`.
const MULTIPLY: BlendComponent = BlendComponent {
    src_factor: BlendFactor::Zero,
    dst_factor: BlendFactor::Src,
    operation: BlendOperation::Add,
};

/// Accumulation pipelines, one per distinct mesh vertex layout, in the same
/// order as [`ExtractedVolumes`] so the pass can walk the two together.
#[derive(Resource, Default)]
pub struct QueuedMomentPipelines(pub Vec<Option<CachedRenderPipelineId>>);

/// The resolve pipeline for one view, keyed on that view's target format.
#[derive(Component, Clone, Copy)]
pub struct MomentResolvePipelineId(pub CachedRenderPipelineId);

/// Which resolve pipeline each view last got, and for which target format, so a
/// view that has not changed does not re-queue one every frame.
type ResolveCache = EntityHashMap<(TextureFormat, CachedRenderPipelineId)>;

#[allow(clippy::too_many_arguments)]
pub fn queue_moment_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<MomentPipelines>,
    mesh_pipeline: Res<MomentMeshPipeline>,
    mut specialized: ResMut<SpecializedMeshPipelines<MomentMeshPipeline>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    volumes: Res<ExtractedVolumes>,
    mut queued: ResMut<QueuedMomentPipelines>,
    views: Query<(Entity, &ExtractedView), (With<MomentTransparency>, With<ExtractedCamera>)>,
    mut cached_resolve: Local<ResolveCache>,
) {
    queued.0.clear();
    for volume in &volumes.0 {
        let id = meshes.get(volume.mesh).and_then(|mesh| {
            specialized
                .specialize(&pipeline_cache, &mesh_pipeline, (), &mesh.layout)
                .map_err(|error| error!("draw: moment pipeline: {error}"))
                .ok()
        });
        queued.0.push(id);
    }

    for (entity, view) in &views {
        let format = view.target_format;
        if let Some((cached_format, id)) = cached_resolve.get(&entity)
            && *cached_format == format
        {
            commands.entity(entity).insert(MomentResolvePipelineId(*id));
            continue;
        }

        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("moment_resolve_pipeline".into()),
            layout: vec![pipelines.resolve_layout.clone()],
            vertex: pipelines.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: pipelines.resolve_shader.clone(),
                targets: vec![Some(ColorTargetState {
                    format,
                    // Multiply. Pure absorption dims what is already there and
                    // adds nothing of its own, so the resolve emits the
                    // transmittance and the blender applies it: `dst = dst * T`.
                    // No alpha bookkeeping and no premultiplication to get
                    // wrong. Emission and in-scattering, when they arrive, are
                    // the terms that will need an additive channel beside this.
                    blend: Some(BlendState {
                        color: MULTIPLY,
                        alpha: MULTIPLY,
                    }),
                    write_mask: ColorWrites::COLOR,
                })],
                ..default()
            }),
            ..default()
        });

        commands.entity(entity).insert(MomentResolvePipelineId(id));
        cached_resolve.insert(entity, (format, id));
    }
}
