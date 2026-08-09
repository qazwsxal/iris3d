//! Volumes, drawn by marching rays through a 3D texture.
//!
//! Deliberately the cheap approach. There is no geometry to speak of: the mesh
//! is a box around the grid, and everything happens in the fragment shader. The
//! field goes to the GPU once, and the isovalue, the step count and the opacity
//! are uniforms — so under the invalidation split in [`super::Dirty`] a slider
//! writes a uniform and rebuilds nothing at all.
//!
//! Three accumulation rules share one ray setup. Maximum and mean ignore the
//! order of the samples, so they need no compositing and no transfer function,
//! which makes them the cheapest correct picture available. Blend does proper
//! front-to-back compositing and takes its colour from the representation's
//! existing [`ColorBy`] map rather than from a transfer-function editor that
//! does not exist.
//!
//! What is missing, and is missing on purpose: empty-space skipping, gradient
//! lighting, and pre-integration. All three are real speedups. None belongs in
//! a first pass.
//!
//! # This does not render correctly yet
//!
//! The image comes out as per-pixel noise instead of a solid volume. What has
//! been ruled out, each by putting an early `return` in the shader and looking
//! at the result:
//!
//! - The box is the right size, in the right place, and rasterises solidly.
//! - The ray-box intersection succeeds for every fragment.
//! - The texture coordinates are right: returning them directly draws the
//!   expected smooth colour cube, dark in the middle and saturated at the
//!   silhouette.
//! - The field itself is right: an x ramp renders purple at low x and yellow at
//!   high x, so the upload, the reordering and the colour maps all work.
//!
//! The noise appears only once the shader reads the texture, and it is there
//! even when the reading loop's result is thrown away and the shader returns a
//! constant with no `discard` on the path. Fragments are therefore producing no
//! output at all, rather than producing a wrong value.
//!
//! Ruled out since:
//!
//! - `textureLoad` in place of `textureSampleLevel`, so no sampler and no rule
//!   about texture access under non-uniform control flow.
//! - The mode branch hoisted out of the loop, so the maximum path has no
//!   conditional `break` before a texture read.
//! - `AlphaMode::Opaque` in place of `Blend`. Identical noise, so the
//!   transparent pass, the blend state and the depth test are all innocent.
//! - A row stride mismatch on upload. The same ramp at 256 samples wide, whose
//!   `R8Unorm` row is the 256 bytes a copy wants, and at 32 wide, whose row is
//!   not, gives the same noise at the same density. A scattered upload would
//!   differ between the two, so the texture layout is sound.
//!
//! What remains is that the values which *do* arrive are correct — the ramp is
//! purple at low x and yellow at high x — while neighbouring invocations
//! produce nothing. Correct-or-absent, per invocation, is not a shape shader
//! logic usually takes. Suspect the pipeline rather than the maths:
//!
//! 1. Run under a different backend, `WGPU_BACKEND=dx12` against `vulkan`. No
//!    rebuild, and it separates a driver problem from ours.
//! 2. Turn MSAA off.
//! 3. Watch for validation output with
//!    `RUST_LOG=wgpu_core=warn,wgpu_hal=warn`.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;

use crate::scene::data::Fields;
use crate::scene::registry::{
    float, text, ParamKind, ParamSpec, RepresentationKind, RepresentationRegistry,
};
use crate::scene::representation::ColorMap;
use crate::scene::dataset::GridData;
use crate::scene::{DataArray, DatasetKind};

use super::{Dirty, Drawable, mark};

const SHADER: &str = "embedded://app/draw/volume.wgsl";

/// How the samples along one ray become one colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeMode {
    /// The largest value along the ray. Order-independent, needs no transfer
    /// function, and shows where the peaks are. The cheapest useful picture.
    Maximum,
    /// The mean value along the ray, which reads like an X-ray.
    Mean,
    /// Front-to-back compositing — true volume rendering.
    Blend,
}

impl VolumeMode {
    fn from_str(name: &str) -> Self {
        match name {
            "mean" => VolumeMode::Mean,
            "blend" => VolumeMode::Blend,
            _ => VolumeMode::Maximum,
        }
    }

    fn index(self) -> f32 {
        match self {
            VolumeMode::Maximum => 0.0,
            VolumeMode::Mean => 1.0,
            VolumeMode::Blend => 2.0,
        }
    }
}

#[derive(Component, Debug, Clone, PartialEq)]
pub struct VolumeStyle {
    /// Which field to draw. Empty means the first scalar, as `ColorBy` does.
    pub field: String,
    pub mode: VolumeMode,
    /// Samples per ray. A quality control, not a brightness control — the
    /// shader scales opacity by the step length so the picture holds still.
    pub steps: f32,
    pub opacity: f32,
}

const MODES: &[&str] = &["maximum", "mean", "blend"];

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "field",
        label: "field",
        kind: ParamKind::Field,
    },
    ParamSpec {
        id: "mode",
        label: "mode",
        kind: ParamKind::Choice {
            options: MODES,
            default: "maximum",
        },
    },
    ParamSpec {
        id: "opacity",
        label: "opacity",
        kind: ParamKind::Float {
            default: 1.0,
            min: 0.01,
            max: 20.0,
            logarithmic: true,
        },
    },
    ParamSpec {
        id: "steps",
        label: "steps",
        kind: ParamKind::Float {
            default: 128.0,
            min: 16.0,
            max: 512.0,
            logarithmic: false,
        },
    },
];

pub fn register(registry: &mut RepresentationRegistry) {
    registry.register(RepresentationKind {
        id: "volume",
        label: "volume",
        supports: |dataset| dataset == DatasetKind::Grid,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(VolumeStyle {
                field: text(params, "field", "").to_string(),
                mode: VolumeMode::from_str(text(params, "mode", "maximum")),
                steps: float(params, "steps", 128.0),
                opacity: float(params, "opacity", 1.0),
            });
        },
    });
}

#[derive(Clone, Copy, ShaderType)]
pub struct VolumeUniform {
    /// World space into the unit cube the field is stored in.
    ///
    /// Computed here rather than inverted in the shader. That is what
    /// `bevy_pbr`'s volumetric fog does, and it means the fragment shader needs
    /// no mesh instance data of its own.
    pub uvw_from_world: Mat4,
    /// `x` steps, `y` opacity, `z` mode, `w` colour map.
    pub options: Vec4,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VolumeMaterial {
    #[uniform(0)]
    pub uniform: VolumeUniform,
    #[texture(1, dimension = "3d")]
    #[sampler(2)]
    pub field: Handle<Image>,
}

impl Material for VolumeMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    /// Blend, which also puts the volume in the transparent pass — so it does
    /// not write depth, and opaque geometry standing inside it still shows.
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

/// The texture currently on the GPU, and which field it came from.
///
/// Kept so a change of mode or opacity does not re-upload tens of megabytes.
/// Only a change of field does.
#[derive(Component, Debug)]
pub struct VolumeTexture {
    field: String,
    image: Handle<Image>,
}

/// Every parameter here is a uniform or the choice of texture, so none of them
/// touches the box mesh.
///
/// Moving the object counts too: `uvw_from_world` is built from the transform,
/// so it goes stale the moment the object moves. This is the one backend where
/// a transform change is a redraw rather than something the scene graph handles
/// on its own.
pub fn invalidate(
    mut commands: Commands,
    changed: Query<
        Entity,
        (
            With<VolumeStyle>,
            Or<(Changed<VolumeStyle>, Changed<GlobalTransform>)>,
        ),
    >,
) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
}

/// The grid's bounding box in local space, wound the ordinary way out.
///
/// An earlier version wound it inside out so that back-face culling handed the
/// shader the ray's *exit* point. That was a trick to avoid specialising the
/// pipeline, and it is not what `bevy_pbr`'s volume rendering does — it keeps
/// `cull_mode: Back` and an outward box. The shader no longer needs the exit
/// point anyway: it takes both ends from the slab test and uses the fragment
/// only for the ray's direction.
///
/// The cost is that the volume vanishes when the camera is inside the box,
/// because every face is then back-facing. Bevy handles that with a second
/// strategy; a first pass can live without one.
fn box_mesh(low: Vec3, high: Vec3) -> Mesh {
    let corners = [
        Vec3::new(low.x, low.y, low.z),
        Vec3::new(high.x, low.y, low.z),
        Vec3::new(high.x, high.y, low.z),
        Vec3::new(low.x, high.y, low.z),
        Vec3::new(low.x, low.y, high.z),
        Vec3::new(high.x, low.y, high.z),
        Vec3::new(high.x, high.y, high.z),
        Vec3::new(low.x, high.y, high.z),
    ];
    // Each face listed counter-clockwise seen from outside.
    let faces: [[u32; 4]; 6] = [
        [0, 1, 2, 3], // back
        [4, 7, 6, 5], // front
        [0, 4, 5, 1], // bottom
        [3, 2, 6, 7], // top
        [0, 3, 7, 4], // left
        [1, 5, 6, 2], // right
    ];

    let mut indices = Vec::with_capacity(36);
    for [a, b, c, d] in faces {
        indices.extend_from_slice(&[a, b, c, a, c, d]);
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        corners.iter().map(|c| [c.x, c.y, c.z]).collect::<Vec<_>>(),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Packs a field into an 8-bit single-channel 3D texture.
///
/// Eight bits rather than the float the client sent, for two reasons. Filtering
/// 32-bit float textures needs a wgpu feature that is not present everywhere,
/// and a 256³ volume is 16 MiB this way against 67 MiB as `f32`. The precision
/// lost is below what the eye resolves through a transfer function.
///
/// Returns the image and the range it was normalised against.
fn field_texture(values: &[f32], grid: &GridData, range: Option<(f32, f32)>) -> Option<Image> {
    let expected = grid.point_count() as usize;
    if values.len() < expected {
        warn!(
            "draw: a volume's field has {} values for a grid of {expected} samples",
            values.len()
        );
        return None;
    }

    let (low, high) = range.unwrap_or_else(|| {
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for value in &values[..expected] {
            if value.is_finite() {
                low = low.min(*value);
                high = high.max(*value);
            }
        }
        (low, high)
    });
    let span = if (high - low).abs() < f32::EPSILON {
        1.0
    } else {
        high - low
    };

    // Reorder while normalising. The wire runs z fastest, which is what a numpy
    // array of shape (x, y, z) gives from a plain `.ravel()`. A 3D texture wants
    // x fastest. Getting this wrong does not fail — it silently transposes the
    // volume, which looks like a plausible render of the wrong thing.
    let (nx, ny, nz) = (
        grid.dims.x as usize,
        grid.dims.y as usize,
        grid.dims.z as usize,
    );
    let mut data = Vec::with_capacity(expected);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let value = values[(x * ny + y) * nz + z];
                data.push((((value - low) / span).clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: grid.dims.x,
            height: grid.dims.y,
            depth_or_array_layers: grid.dims.z,
        },
        TextureDimension::D3,
        data,
        TextureFormat::R8Unorm,
        // The CPU copy is dead weight once uploaded; this is rebuilt from the
        // `DataArray` whenever it changes.
        RenderAssetUsages::RENDER_WORLD,
    );
    // Linear filtering is the whole reason for the 8-bit format.
    image.sampler = ImageSampler::linear();
    Some(image)
}

fn colour_map_index(map: ColorMap) -> f32 {
    match map {
        ColorMap::Viridis => 0.0,
        ColorMap::CoolWarm => 1.0,
        // The shader has no element colouring, and a volume has no elements.
        ColorMap::Grayscale | ColorMap::ByElement => 2.0,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn draw_volumes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VolumeMaterial>>,
    mut images: ResMut<Assets<Image>>,
    arrays: Res<Assets<DataArray>>,
    dirty: Query<Drawable<VolumeStyle, VolumeMaterial>>,
    cached: Query<&VolumeTexture>,
    placements: Query<&GlobalTransform>,
    grids: Query<(&GridData, Option<&Fields>)>,
) {
    for (entity, style, colour, _subset, source, dirty, mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Ok((grid, fields)) = grids.get(source.0) else {
            continue;
        };
        let Ok(placement) = placements.get(entity) else {
            continue;
        };

        // Named field first, then the first scalar in name order — the same
        // fallback `ColorBy` uses, so "auto" means the same thing everywhere.
        let Some(fields) = fields else { continue };
        let name = if style.field.is_empty() {
            let mut scalars: Vec<&String> = fields
                .0
                .iter()
                .filter(|(_, field)| field.kind == crate::scene::data::FieldKind::Scalar)
                .map(|(name, _)| name)
                .collect();
            scalars.sort();
            match scalars.first() {
                Some(name) => (*name).clone(),
                None => continue,
            }
        } else {
            style.field.clone()
        };
        let Some(field) = fields.0.get(&name) else {
            warn!("draw: a volume names field \"{name}\", which the grid does not have");
            continue;
        };

        // Re-uploading is the one expensive thing here, so it happens only when
        // the geometry is stale or the chosen field actually changed.
        let previous = cached.get(entity).ok();
        let reusable = previous.filter(|cache| cache.field == name && !dirty.geometry);
        let image = match reusable {
            Some(cache) => cache.image.clone(),
            None => {
                let Some(array) = arrays.get(&field.array) else {
                    continue;
                };
                let Some(texture) = field_texture(&array.to_f32(), grid, colour.range) else {
                    continue;
                };
                let handle = images.add(texture);
                commands.entity(entity).insert(VolumeTexture {
                    field: name.clone(),
                    image: handle.clone(),
                });
                handle
            }
        };

        let size = (grid.dims.saturating_sub(UVec3::ONE)).as_vec3() * grid.spacing;
        if dirty.geometry {
            super::ensure_mesh(
                &mut commands,
                entity,
                &mut meshes,
                mesh3d,
                box_mesh(grid.origin, grid.origin + size),
            );
        }

        // World -> local -> unit cube, composed once here so the shader needs
        // no inverse and no mesh instance data.
        let uvw_from_world = Mat4::from_scale(1.0 / size.max(Vec3::splat(f32::EPSILON)))
            * Mat4::from_translation(-grid.origin)
            * placement.to_matrix().inverse();

        super::ensure_material(
            &mut commands,
            entity,
            &mut materials,
            material3d,
            VolumeMaterial {
                uniform: VolumeUniform {
                    uvw_from_world,
                    options: Vec4::new(
                        style.steps,
                        style.opacity,
                        style.mode.index(),
                        colour_map_index(colour.map),
                    ),
                },
                field: image,
            },
        );

        debug!(
            "draw: volume over \"{name}\", {:?} of {} samples",
            style.mode,
            grid.point_count()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(dims: [u32; 3]) -> GridData {
        GridData {
            origin: Vec3::ZERO,
            spacing: Vec3::ONE,
            dims: UVec3::from(dims),
        }
    }

    #[test]
    fn the_box_wraps_the_grid() {
        let mesh = box_mesh(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(mesh.count_vertices(), 8);
        assert_eq!(mesh.indices().map(|i| i.len()), Some(36));
    }

    #[test]
    fn a_field_becomes_an_eight_bit_texture() {
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = field_texture(&values, &grid([2, 2, 2]), None).expect("built");
        assert_eq!(image.texture_descriptor.format, TextureFormat::R8Unorm);
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 2);

        // The range is normalised across the data, so the ends are the extremes.
        let data = image.data.expect("kept the pixels");
        assert_eq!(data.len(), 8);
        assert_eq!(data[0], 0);
        assert_eq!(data[7], 255);
    }

    /// The wire runs z fastest and a 3D texture runs x fastest. A field that
    /// varies only along x must therefore come out varying along the texture's
    /// *first* axis. Getting this wrong transposes the volume silently, so this
    /// is the test that catches it.
    #[test]
    fn the_texture_is_reordered_from_wire_order() {
        // values[x][y][z] = x, laid out z fastest.
        let mut values = Vec::new();
        for x in 0..2 {
            for _y in 0..2 {
                for _z in 0..2 {
                    values.push(x as f32);
                }
            }
        }

        let image = field_texture(&values, &grid([2, 2, 2]), None).expect("built");
        let data = image.data.expect("kept the pixels");
        // Texture order is x fastest, so neighbouring pairs must differ.
        assert_eq!(data[0], 0, "x = 0");
        assert_eq!(data[1], 255, "x = 1");
        assert_eq!(data[2], 0, "x = 0 of the next row");
        assert_eq!(data[3], 255, "x = 1 of the next row");
    }

    /// An explicit colour range wins over the data's own, so two volumes can be
    /// compared against the same scale.
    #[test]
    fn an_explicit_range_overrides_the_data() {
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = field_texture(&values, &grid([2, 2, 2]), Some((0.0, 14.0))).expect("built");
        let data = image.data.expect("kept the pixels");
        // 7 of 14 is half way, not the top.
        assert!((120..=136).contains(&data[7]), "got {}", data[7]);
    }

    /// A field that does not cover the grid is refused rather than sampled off
    /// the end of its own array.
    #[test]
    fn a_short_field_is_refused() {
        assert!(field_texture(&[0.0, 1.0], &grid([4, 4, 4]), None).is_none());
    }
}
