//! The two pipelines: accumulate, then reconstruct.

use bevy::asset::load_embedded_asset;
use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy::ecs::entity::EntityHashMap;
use bevy::mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, texture_2d_multisampled, texture_3d,
    texture_depth_2d, texture_depth_2d_multisampled, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation,
    BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
    DepthStencilState, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
    RenderPipelineDescriptor, SamplerBindingType, ShaderStages, SpecializedMeshPipeline,
    SpecializedMeshPipelineError, SpecializedMeshPipelines, SpecializedRenderPipeline,
    SpecializedRenderPipelines, TextureFormat, TextureSampleType, VertexState,
};
use bevy::render::view::{ExtractedView, Msaa, ViewUniform};
use bevy::shader::Shader;

use super::MomentView;
use super::extract::{ExtractedGrids, ExtractedVolumes, ShellLighting};
use super::prepare::{GridUniform, MomentBounds};

/// The moment target's format, fixed here so the pipeline and the texture
/// cannot disagree.
pub const MOMENT_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

/// Whether a view is multisampled, which every layout and pipeline here has to
/// agree with.
///
/// A sample count is not a uniform: it changes the *type* of the depth and
/// moment bindings, so each layout exists in two versions and the shaders are
/// compiled with `MULTISAMPLED` defined or not. Both are built at startup
/// rather than on demand, because there are exactly two and the alternative is
/// a cache to get wrong.
pub fn multisampled(samples: u32) -> bool {
    samples > 1
}

/// Bind group 0 of the accumulation pass.
fn moment_layout(multisampled: bool) -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        if multisampled {
            "moment_bind_group_layout_multisampled"
        } else {
            "moment_bind_group_layout"
        },
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                storage_buffer_read_only_sized(false, None),
                if multisampled {
                    texture_depth_2d_multisampled()
                } else {
                    texture_depth_2d()
                },
                uniform_buffer::<MomentBounds>(false),
            ),
        ),
    )
}

/// Bind group 0 of the resolve pass.
///
/// The moment targets take the view's sample count too, so all three textures
/// change type together.
fn resolve_layout(multisampled: bool) -> BindGroupLayoutDescriptor {
    // Never filtered, and loaded by integer coordinate. Interpolating moments
    // between neighbouring pixels is only meaningful in a light-space map,
    // where prefiltering is the whole advantage; in view space it mixes
    // unrelated depth distributions.
    let unfiltered = TextureSampleType::Float { filterable: false };
    let moments = || {
        if multisampled {
            texture_2d_multisampled(unfiltered)
        } else {
            texture_2d(unfiltered)
        }
    };
    BindGroupLayoutDescriptor::new(
        if multisampled {
            "moment_resolve_bind_group_layout_multisampled"
        } else {
            "moment_resolve_bind_group_layout"
        },
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                moments(),
                moments(),
                if multisampled {
                    texture_depth_2d_multisampled()
                } else {
                    texture_depth_2d()
                },
                uniform_buffer::<MomentBounds>(false),
            ),
        ),
    )
}

/// Bind group 0 of the shell pass.
///
/// The moments are here because a reflection has to be dimmed by whatever
/// medium stands in front of it, and that is the one question only the moments
/// can answer — the totals alone would attenuate a near highlight by the far
/// wall's absorbance as well as its own.
fn shell_layout(multisampled: bool) -> BindGroupLayoutDescriptor {
    let unfiltered = TextureSampleType::Float { filterable: false };
    let moments = || {
        if multisampled {
            texture_2d_multisampled(unfiltered)
        } else {
            texture_2d(unfiltered)
        }
    };
    BindGroupLayoutDescriptor::new(
        if multisampled {
            "moment_shell_bind_group_layout_multisampled"
        } else {
            "moment_shell_bind_group_layout"
        },
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                storage_buffer_read_only_sized(false, None),
                uniform_buffer::<ShellLighting>(false),
                moments(),
                moments(),
                uniform_buffer::<MomentBounds>(false),
            ),
        ),
    )
}

/// Bind group 1 of both grid passes: the grid's uniform, its field, and its
/// colour ramp.
///
/// Independent of the sample count, unlike every other layout here — these are
/// the grid's own assets rather than view-sized targets, so nothing about them
/// changes when MSAA does.
///
/// Both textures are declared filterable, and that is load-bearing rather than
/// incidental: hardware filtering on the field is the entire mechanism behind
/// "nearest neighbour versus linear is a sampler setting". Neither shader picks
/// a reconstruction filter; the `ImageSampler` on the asset does.
fn grid_layout() -> BindGroupLayoutDescriptor {
    let filterable = TextureSampleType::Float { filterable: true };
    BindGroupLayoutDescriptor::new(
        "moment_grid_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<GridUniform>(false),
                texture_3d(filterable),
                sampler(SamplerBindingType::Filtering),
                texture_2d(filterable),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    )
}

#[derive(Resource)]
pub struct MomentPipelines {
    moment_layouts: [BindGroupLayoutDescriptor; 2],
    resolve_layouts: [BindGroupLayoutDescriptor; 2],
    shell_layouts: [BindGroupLayoutDescriptor; 2],
    grid_layout: BindGroupLayoutDescriptor,
    pub resolve_shader: Handle<Shader>,
    pub fullscreen: FullscreenShader,
}

impl MomentPipelines {
    pub fn moment_layout(&self, samples: u32) -> &BindGroupLayoutDescriptor {
        &self.moment_layouts[usize::from(multisampled(samples))]
    }

    pub fn resolve_layout(&self, samples: u32) -> &BindGroupLayoutDescriptor {
        &self.resolve_layouts[usize::from(multisampled(samples))]
    }

    pub fn shell_layout(&self, samples: u32) -> &BindGroupLayoutDescriptor {
        &self.shell_layouts[usize::from(multisampled(samples))]
    }

    pub fn grid_layout(&self) -> &BindGroupLayoutDescriptor {
        &self.grid_layout
    }
}

/// The two grid pipelines: accumulate into the moment target, then emit into the
/// view target.
///
/// Plain [`SpecializedRenderPipeline`]s rather than mesh ones, because a grid
/// has no geometry. Both draw a fullscreen triangle — see the header of
/// `volume.wgsl` for why a bounding box is the wrong shape for this — so there
/// is no vertex layout to specialise over and the key is only what the targets
/// demand.
#[derive(Resource)]
pub struct GridPipelines {
    moment_layouts: [BindGroupLayoutDescriptor; 2],
    resolve_layouts: [BindGroupLayoutDescriptor; 2],
    grid_layout: BindGroupLayoutDescriptor,
    accumulate_shader: Handle<Shader>,
    emit_shader: Handle<Shader>,
    fullscreen: FullscreenShader,
}

/// Which grid pipeline is wanted, and what it has to match.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridKey {
    pub samples: u32,
    /// The view target's format. Only the emission pass uses it — the
    /// accumulation writes into [`MOMENT_FORMAT`], which is fixed.
    pub format: TextureFormat,
    pub emit: bool,
}

impl SpecializedRenderPipeline for GridPipelines {
    type Key = GridKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let multisampled = multisampled(key.samples);
        let shader_defs = if multisampled {
            vec!["MULTISAMPLED".into()]
        } else {
            vec![]
        };

        // The emission pass adds light to the view target; the accumulation adds
        // absorbance to the moment target. Both are additive, and for the same
        // reason: addition commutes, so neither needs anything sorted.
        let targets = if key.emit {
            vec![Some(ColorTargetState {
                format: key.format,
                blend: Some(BlendState {
                    color: ADDITIVE,
                    alpha: ADDITIVE,
                }),
                write_mask: ColorWrites::COLOR,
            })]
        } else {
            let moment_target = || {
                Some(ColorTargetState {
                    format: MOMENT_FORMAT,
                    blend: Some(BlendState {
                        color: ADDITIVE,
                        alpha: ADDITIVE,
                    }),
                    write_mask: ColorWrites::ALL,
                })
            };
            vec![moment_target(), moment_target()]
        };

        // Group 0 is borrowed wholesale rather than declared again: the
        // accumulation wants exactly what `moment.wgsl` gets, and the emission
        // wants exactly what `resolve.wgsl` gets. A shader may leave a binding in
        // the layout undeclared, which is what lets the grid passes ignore the
        // mesh instance buffer without a layout of their own.
        let view_layout = if key.emit {
            self.resolve_layouts[usize::from(multisampled)].clone()
        } else {
            self.moment_layouts[usize::from(multisampled)].clone()
        };

        let shader = if key.emit {
            self.emit_shader.clone()
        } else {
            self.accumulate_shader.clone()
        };

        RenderPipelineDescriptor {
            label: Some(if key.emit {
                "moment_grid_emit_pipeline".into()
            } else {
                "moment_grid_pipeline".into()
            }),
            layout: vec![view_layout, self.grid_layout.clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader,
                shader_defs,
                targets,
                ..default()
            }),
            // No depth attachment on either. The accumulation clamps against the
            // depth buffer in the shader rather than testing, and the emission
            // does the same — a volume's contribution has to be *truncated* at an
            // occluder, not rejected by it.
            multisample: MultisampleState {
                count: key.samples,
                ..default()
            },
            ..default()
        }
    }
}

pub fn init_moment_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen: Res<FullscreenShader>,
) {
    let moment_shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "moment.wgsl");

    commands.insert_resource(MomentMeshPipeline {
        layouts: [moment_layout(false), moment_layout(true)],
        shader: moment_shader.clone(),
    });
    commands.insert_resource(ShellMeshPipeline {
        layouts: [shell_layout(false), shell_layout(true)],
        shader: load_embedded_asset!(asset_server.as_ref(), "shell.wgsl"),
    });
    commands.insert_resource(GridPipelines {
        moment_layouts: [moment_layout(false), moment_layout(true)],
        resolve_layouts: [resolve_layout(false), resolve_layout(true)],
        grid_layout: grid_layout(),
        accumulate_shader: load_embedded_asset!(asset_server.as_ref(), "volume.wgsl"),
        emit_shader: load_embedded_asset!(asset_server.as_ref(), "emit.wgsl"),
        fullscreen: fullscreen.clone(),
    });
    commands.insert_resource(MomentPipelines {
        moment_layouts: [moment_layout(false), moment_layout(true)],
        resolve_layouts: [resolve_layout(false), resolve_layout(true)],
        shell_layouts: [shell_layout(false), shell_layout(true)],
        grid_layout: grid_layout(),
        resolve_shader: load_embedded_asset!(asset_server.as_ref(), "resolve.wgsl"),
        fullscreen: fullscreen.clone(),
    });
}

/// The grid pipelines queued for each view, keyed the same way the mesh ones
/// are.
#[derive(Resource, Default)]
pub struct QueuedGridPipelines(pub EntityHashMap<(CachedRenderPipelineId, CachedRenderPipelineId)>);

/// Queues both grid pipelines for every view that has grids to draw.
///
/// Keyed on the view rather than cached globally because the emission pass
/// writes into the view's own target, whose format is the view's business.
pub fn queue_grid_pipelines(
    mut queued: ResMut<QueuedGridPipelines>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GridPipelines>>,
    pipeline: Res<GridPipelines>,
    pipeline_cache: Res<PipelineCache>,
    grids: Res<ExtractedGrids>,
    views: Query<(Entity, &ExtractedView, &Msaa), With<MomentView>>,
) {
    queued.0.clear();
    if grids.0.is_empty() {
        return;
    }
    for (entity, view, msaa) in &views {
        let format = view.target_format;
        let samples = msaa.samples();
        let accumulate = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            GridKey {
                samples,
                format,
                emit: false,
            },
        );
        let emit = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            GridKey {
                samples,
                format,
                emit: true,
            },
        );
        queued.0.insert(entity, (accumulate, emit));
    }
}

/// What the shell pipeline has to be specialised over.
///
/// Unlike the accumulation, this one draws into the *view's* target rather than
/// a target of its own, so the format is not a constant it can assume.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShellKey {
    pub samples: u32,
    pub format: TextureFormat,
}

/// Specialises the shell pipeline over the mesh's vertex layout.
///
/// Position and normal, which is the difference from the accumulation: a
/// reflection is about which way a surface faces, where an absorbance is only
/// about where its boundary is.
#[derive(Resource)]
pub struct ShellMeshPipeline {
    layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
}

impl SpecializedMeshPipeline for ShellMeshPipeline {
    type Key = ShellKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        ])?;
        let multisampled = multisampled(key.samples);
        let shader_defs = if multisampled {
            vec!["MULTISAMPLED".into()]
        } else {
            vec![]
        };

        Ok(RenderPipelineDescriptor {
            label: Some("moment_shell_pipeline".into()),
            layout: vec![self.layouts[usize::from(multisampled)].clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.format,
                    // Additive, so the shell adds reflected light and takes
                    // nothing away — and so the two faces of a closed mesh
                    // compose without being sorted, exactly as the accumulation
                    // does.
                    blend: Some(BlendState {
                        color: ADDITIVE,
                        alpha: ADDITIVE,
                    }),
                    write_mask: ColorWrites::COLOR,
                })],
                ..default()
            }),
            // Both faces. A thin shell reflects at its far interface too, and
            // that second rim is much of what makes a shape read as hollow.
            primitive: PrimitiveState {
                cull_mode: None,
                ..default()
            },
            // Depth *tested*, unlike the accumulation, and this is the one place
            // in the pathway where that is right: a shell fragment is an
            // ordinary sample of a surface rather than one endpoint of an
            // interval, so hiding it behind opaque geometry loses nothing.
            // `GreaterEqual` because Bevy's projection is reverse-Z, and no
            // depth write because the shell must not occlude anything.
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: default(),
                bias: default(),
            }),
            multisample: MultisampleState {
                count: key.samples,
                ..default()
            },
            ..default()
        })
    }
}

/// Specialises the accumulation pipeline over the mesh's vertex layout.
///
/// Only the position is pulled. The moment pass has no lighting, no colour and
/// no texturing, so normals and vertex colours are dead weight in it — but the
/// meshes are built by an actor kind that may carry them, so the stride still
/// has to be read off the real layout rather than assumed.
#[derive(Resource)]
pub struct MomentMeshPipeline {
    layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
}

impl SpecializedMeshPipeline for MomentMeshPipeline {
    /// The view's sample count. The moment target matches the view, so the
    /// pipeline writing into it has to as well.
    type Key = u32;

    fn specialize(
        &self,
        samples: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        let multisampled = multisampled(samples);
        let shader_defs = if multisampled {
            vec!["MULTISAMPLED".into()]
        } else {
            vec![]
        };

        Ok(RenderPipelineDescriptor {
            label: Some("moment_pipeline".into()),
            layout: vec![self.layouts[usize::from(multisampled)].clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                // Both targets additive. This is the one line that makes the
                // whole method order-independent.
                targets: vec![
                    Some(ColorTargetState {
                        format: MOMENT_FORMAT,
                        blend: Some(BlendState {
                            color: ADDITIVE,
                            alpha: ADDITIVE,
                        }),
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: MOMENT_FORMAT,
                        blend: Some(BlendState {
                            color: ADDITIVE,
                            alpha: ADDITIVE,
                        }),
                        write_mask: ColorWrites::ALL,
                    }),
                ],
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
            // Matches the moment target, which matches the view. Coverage is
            // what antialiases a volume's silhouette: a fragment's contribution
            // reaches only the samples it covers, so a half-covered edge pixel
            // accumulates half the absorbance without the shader running twice.
            multisample: MultisampleState {
                count: samples,
                ..default()
            },
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

/// Accumulation pipelines, in the same order as [`ExtractedVolumes`] so the
/// pass can walk the two together — but one such list per sample count.
///
/// Keyed by sample count rather than a single list, because two views drawing
/// the same volumes can be multisampled differently: iris3d's viewport and a
/// camera rendering to an image need not agree, and a pipeline built for one
/// cannot write into the other's target. The pass looks up its own view's
/// count.
#[derive(Resource, Default)]
pub struct QueuedMomentPipelines(pub HashMap<u32, Vec<Option<CachedRenderPipelineId>>>);

/// Shell pipelines, in the same order as [`ExtractedVolumes`], one list per
/// distinct target the shell is drawn into.
///
/// `None` where a volume has no shell as well as where one could not be built,
/// which the pass treats the same way: skip it.
#[derive(Resource, Default)]
pub struct QueuedShellPipelines(pub HashMap<ShellKey, Vec<Option<CachedRenderPipelineId>>>);

/// The resolve pipeline for one view, keyed on that view's target format.
#[derive(Component, Clone, Copy)]
pub struct MomentResolvePipelineId(pub CachedRenderPipelineId);

/// Which resolve pipeline each view last got, and for which target format and
/// sample count, so a view that has not changed does not re-queue one every
/// frame.
type ResolveCache = EntityHashMap<(TextureFormat, u32, CachedRenderPipelineId)>;

/// The views a resolve pipeline is queued for: this pathway's, and only those
/// that are real cameras.
type MomentViews = (With<MomentView>, With<ExtractedCamera>);

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
    views: Query<(Entity, &ExtractedView, &Msaa), MomentViews>,
    mut cached_resolve: Local<ResolveCache>,
) {
    // One accumulation pipeline per volume per distinct sample count on screen.
    // Usually that is one count and the inner loop runs once.
    queued.0.clear();
    for samples in views.iter().map(|(_, _, msaa)| msaa.samples()) {
        if queued.0.contains_key(&samples) {
            continue;
        }
        let ids = volumes
            .0
            .iter()
            .map(|volume| {
                meshes.get(volume.mesh).and_then(|mesh| {
                    specialized
                        .specialize(&pipeline_cache, &mesh_pipeline, samples, &mesh.layout)
                        .map_err(|error| error!("draw: moment pipeline: {error}"))
                        .ok()
                })
            })
            .collect();
        queued.0.insert(samples, ids);
    }

    for (entity, view, msaa) in &views {
        let format = view.target_format;
        let samples = msaa.samples();
        if let Some((cached_format, cached_samples, id)) = cached_resolve.get(&entity)
            && *cached_format == format
            && *cached_samples == samples
        {
            commands.entity(entity).insert(MomentResolvePipelineId(*id));
            continue;
        }

        let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("moment_resolve_pipeline".into()),
            layout: vec![pipelines.resolve_layout(samples).clone()],
            vertex: pipelines.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: pipelines.resolve_shader.clone(),
                shader_defs: if multisampled(samples) {
                    vec!["MULTISAMPLED".into()]
                } else {
                    vec![]
                },
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
            // The resolve writes into the view's own target, so it is
            // multisampled exactly when the view is. The shader reads
            // `@builtin(sample_index)` in that case, which makes it run per
            // sample and keeps each sample's own transmittance.
            multisample: MultisampleState {
                count: samples,
                ..default()
            },
            ..default()
        });

        commands.entity(entity).insert(MomentResolvePipelineId(id));
        cached_resolve.insert(entity, (format, samples, id));
    }
}

/// Builds a shell pipeline for every volume that asked for one.
///
/// Separate from [`queue_moment_pipelines`] because the two are specialised
/// over different things — the shell needs the view's target format, the
/// accumulation writes into a target of its own — and because most volumes have
/// no shell at all, so this loop usually does nothing.
#[allow(clippy::too_many_arguments)]
pub fn queue_shell_pipelines(
    pipeline_cache: Res<PipelineCache>,
    shell_pipeline: Res<ShellMeshPipeline>,
    mut specialized: ResMut<SpecializedMeshPipelines<ShellMeshPipeline>>,
    meshes: Res<RenderAssets<RenderMesh>>,
    volumes: Res<ExtractedVolumes>,
    mut queued: ResMut<QueuedShellPipelines>,
    views: Query<(&ExtractedView, &Msaa), MomentViews>,
) {
    queued.0.clear();
    if !volumes.0.iter().any(|volume| volume.shell.is_some()) {
        return;
    }

    for (view, msaa) in &views {
        let key = ShellKey {
            samples: msaa.samples(),
            format: view.target_format,
        };
        if queued.0.contains_key(&key) {
            continue;
        }
        let ids = volumes
            .0
            .iter()
            .map(|volume| {
                volume.shell?;
                let mesh = meshes.get(volume.mesh)?;
                specialized
                    .specialize(&pipeline_cache, &shell_pipeline, key, &mesh.layout)
                    .map_err(|error| {
                        // The likely one: a mesh with no normals. Named rather
                        // than passed through, because "missing vertex
                        // attribute NORMAL" does not say which of the two
                        // passes wanted it or why.
                        error!("draw: moment shell pipeline: {error}");
                    })
                    .ok()
            })
            .collect();
        queued.0.insert(key, ids);
    }
}
