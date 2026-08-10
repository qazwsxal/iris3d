//! Volumes, drawn by marching rays through a 3D texture.
//!
//! Deliberately the cheap approach. There is no geometry to speak of: the mesh
//! is a box around the grid, and everything happens in the fragment shader. The
//! field goes to the GPU once, and the isovalue, the step count and the opacity
//! are uniforms — so under the invalidation split in [`super::Dirty`] a slider
//! writes a uniform and rebuilds nothing at all.
//!
//! Three accumulation rules share one ray setup. Maximum and mean ignore the
//! order of the samples, so they need no compositing at all, which makes them
//! the cheapest correct picture available. Blend does proper front-to-back
//! compositing.
//!
//! Density and colour are separate choices. The density decides how solid the
//! volume is; [`ColorBy`](crate::scene::ColorBy) decides what tints it, and may
//! name a different field — density from one quantity and colour from another
//! is the usual pairing in scientific volume rendering. Both live in one
//! two-channel texture, so a step still costs one sample.
//!
//! What is missing, and is missing on purpose: empty-space skipping, gradient
//! lighting, and pre-integration. All three are real speedups. None belongs in
//! a first pass. Nor is there a transfer function worth the name — colour is a
//! fixed ramp and opacity is the density times a scale, rather than two curves
//! you can bend.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat,
};
use bevy::shader::ShaderRef;

use crate::scene::actor::ColorMap;
use crate::scene::registry::{
    ActorKind, ActorRegistry, ParamKind, ParamSpec, float, text, uvec3 as param_uvec3,
    vec3 as param_vec3,
};
use crate::scene::{DataArray, DataStore};

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
    pub mode: VolumeMode,
    /// Samples per ray. A quality control, not a brightness control — the
    /// shader scales opacity by the step length so the picture holds still.
    pub steps: f32,
    pub opacity: f32,
}

/// Where the samples sit, kept apart from [`VolumeStyle`] on purpose.
///
/// Style changes are material-only: opacity and steps go to the shader as
/// uniforms and touch nothing else. The grid is the opposite — it decides the
/// box mesh and the texture upload. Sharing one component would mean a single
/// frame of an opacity drag rebuilding the mesh and re-uploading a 64³ texture,
/// which is exactly what grading `Dirty` exists to avoid.
///
/// This is the same reasoning that keeps `Bindings` separate.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct VolumeGrid {
    pub origin: Vec3,
    pub spacing: Vec3,
    pub dims: UVec3,
}

impl VolumeGrid {
    /// Samples in the grid.
    pub fn point_count(&self) -> u64 {
        self.dims.x as u64 * self.dims.y as u64 * self.dims.z as u64
    }

    /// Extent of the box the samples span. An axis of one sample spans nothing,
    /// which is what a slice uploaded as a grid looks like — hence the
    /// saturating subtraction rather than one that wraps.
    pub fn size(&self) -> Vec3 {
        self.dims.saturating_sub(UVec3::ONE).as_vec3() * self.spacing
    }
}

const MODES: &[&str] = &["maximum", "mean", "blend"];

const PARAMS: &[ParamSpec] = &[
    // What makes the volume solid. One value per sample; the grid says how those
    // samples are arranged.
    ParamSpec {
        id: "density",
        label: "density",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: true,
        },
    },
    // What tints it, which is a separate choice from what makes it solid.
    // Density from one quantity and colour from another is the usual pairing in
    // scientific volume rendering; unbound means colour by the density itself.
    ParamSpec {
        id: "colour",
        label: "colour by",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
        },
    },
    // The geometry a grid does not upload. This is the whole reason the grid
    // type exists: 64³ samples state their arrangement in nine numbers rather
    // than 262144 coordinates, so it cannot arrive as an array.
    ParamSpec {
        id: "dims",
        label: "samples",
        kind: ParamKind::Vector {
            components: 3,
            default: &[1.0, 1.0, 1.0],
            min: 1.0,
            max: 4096.0,
            integral: true,
        },
    },
    ParamSpec {
        id: "origin",
        label: "origin",
        kind: ParamKind::Vector {
            components: 3,
            default: &[0.0, 0.0, 0.0],
            min: -1.0e6,
            max: 1.0e6,
            integral: false,
        },
    },
    ParamSpec {
        id: "spacing",
        label: "spacing",
        kind: ParamKind::Vector {
            components: 3,
            default: &[1.0, 1.0, 1.0],
            min: 1.0e-6,
            max: 1.0e6,
            integral: false,
        },
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

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "volume",
        label: "volume",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert((
                VolumeStyle {
                    mode: VolumeMode::from_str(text(params, "mode", "maximum")),
                    steps: float(params, "steps", 128.0),
                    opacity: float(params, "opacity", 1.0),
                },
                VolumeGrid {
                    origin: param_vec3(params, "origin", Vec3::ZERO),
                    spacing: param_vec3(params, "spacing", Vec3::ONE),
                    dims: param_uvec3(params, "dims", UVec3::ONE),
                },
            ));
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
    /// The arrays this was built from, by handle. Keyed on the binding rather
    /// than on a field name, so rebinding is what invalidates it.
    density: u64,
    /// `None` when colour comes from the density.
    colour: Option<u64>,
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
    regridded: Query<Entity, Changed<VolumeGrid>>,
) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
    // The grid decides the box mesh and the texture, so moving or resizing it is
    // a rebuild rather than a uniform write.
    for entity in &regridded {
        mark(&mut commands, entity, Dirty::GEOMETRY);
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

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        corners.iter().map(|c| [c.x, c.y, c.z]).collect::<Vec<_>>(),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Lowest and highest finite value.
fn range_of(values: &[f32]) -> (f32, f32) {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            low = low.min(*value);
            high = high.max(*value);
        }
    }
    (low, high)
}

fn quantise(value: f32, (low, high): (f32, f32)) -> u8 {
    let span = if (high - low).abs() < f32::EPSILON {
        1.0
    } else {
        high - low
    };
    (((value - low) / span).clamp(0.0, 1.0) * 255.0) as u8
}

/// Packs the density and the colour field into one two-channel 3D texture.
///
/// Two channels rather than two textures, so a step still costs one sample.
/// Red is the density, which decides opacity; green is whatever the actor is
/// coloured by. When they are the same field both channels hold it, which
/// costs a byte per sample and keeps the shader free of a special case.
///
/// Eight bits each rather than the float the client sent, for two reasons.
/// Filtering 32-bit float textures needs a wgpu feature that is not present
/// everywhere, and a 256³ volume is 33 MiB this way against 134 MiB as two
/// `f32` channels. The precision lost is below what the eye resolves through a
/// colour ramp.
///
/// The two channels are scaled independently: density always against its own
/// range, colour against `ColorBy::range` when one is set. That is what a
/// colour range means everywhere else, and it lets two volumes be compared
/// against the same scale without changing how solid either one is.
fn volume_texture(
    density: &[f32],
    colour: Option<&[f32]>,
    grid: &VolumeGrid,
    colour_range: Option<(f32, f32)>,
) -> Option<Image> {
    let expected = grid.point_count() as usize;
    if density.len() < expected {
        warn!(
            "draw: a volume's density field has {} values for a grid of {expected} samples",
            density.len()
        );
        return None;
    }
    // A colour field that does not cover the grid is dropped rather than
    // fataled: the volume still has a shape worth seeing.
    let colour = colour.filter(|values| {
        let long_enough = values.len() >= expected;
        if !long_enough {
            warn!("draw: a volume's colour field is shorter than its grid; colouring by density");
        }
        long_enough
    });

    let density_range = range_of(&density[..expected]);
    let colour_scale =
        colour.map(|values| colour_range.unwrap_or_else(|| range_of(&values[..expected])));

    // Reorder while normalising. The wire runs z fastest, which is what a numpy
    // array of shape (x, y, z) gives from a plain `.ravel()`. A 3D texture wants
    // x fastest. Getting this wrong does not fail — it silently transposes the
    // volume, which looks like a plausible render of the wrong thing.
    let (nx, ny, nz) = (
        grid.dims.x as usize,
        grid.dims.y as usize,
        grid.dims.z as usize,
    );
    let mut data = Vec::with_capacity(expected * 2);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = (x * ny + y) * nz + z;
                let red = quantise(density[index], density_range);
                data.push(red);
                data.push(match (colour, colour_scale) {
                    (Some(values), Some(scale)) => quantise(values[index], scale),
                    // No separate colour field, so colour by the density.
                    _ => red,
                });
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
        TextureFormat::Rg8Unorm,
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
    store: Res<DataStore>,
    dirty: Query<Drawable<VolumeStyle, VolumeMaterial>>,
    cached: Query<&VolumeTexture>,
    placements: Query<&GlobalTransform>,
    grids: Query<&VolumeGrid>,
) {
    for (entity, style, colour, _subset, bound, dirty, mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let (Ok(grid), Ok(placement)) = (grids.get(entity), placements.get(entity)) else {
            continue;
        };

        // Required, so `check_bindings` refused an actor without it.
        let Some(density_handle) = bound.get("density") else {
            continue;
        };
        // Colouring by the same array as the density is the same as not naming
        // one, so it is dropped rather than uploaded into both channels.
        let colour_handle = bound.get("colour").filter(|id| *id != density_handle);

        // Re-uploading is the one expensive thing here, so it happens only when
        // the geometry is stale or one of the bound arrays actually changed.
        let previous = cached.get(entity).ok();
        let reusable = previous.filter(|cache| {
            cache.density == density_handle && cache.colour == colour_handle && !dirty.geometry
        });
        let image = match reusable {
            Some(cache) => cache.image.clone(),
            None => {
                let Some(density) = super::bound(bound, "density", &store, &arrays) else {
                    continue;
                };
                let tint = colour_handle
                    .and_then(|id| store.get(id))
                    .and_then(|held| arrays.get(&held.handle))
                    .map(|array| array.to_f32());
                let Some(texture) =
                    volume_texture(&density.to_f32(), tint.as_deref(), grid, colour.range)
                else {
                    continue;
                };
                let handle = images.add(texture);
                commands.entity(entity).insert(VolumeTexture {
                    density: density_handle,
                    colour: colour_handle,
                    image: handle.clone(),
                });
                handle
            }
        };

        let size = grid.size();
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
            "draw: volume, density d{density_handle} coloured by d{}, {:?} of {} samples",
            colour_handle.unwrap_or(density_handle),
            style.mode,
            grid.point_count()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(dims: [u32; 3]) -> VolumeGrid {
        VolumeGrid {
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
    fn a_field_becomes_a_two_channel_texture() {
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = volume_texture(&values, None, &grid([2, 2, 2]), None).expect("built");
        assert_eq!(image.texture_descriptor.format, TextureFormat::Rg8Unorm);
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 2);

        // Two bytes per sample, and the range is normalised across the data, so
        // the ends are the extremes.
        let data = image.data.expect("kept the pixels");
        assert_eq!(data.len(), 16);
        assert_eq!(data[0], 0);
        assert_eq!(data[14], 255);
    }

    /// Without a colour field, green repeats red — so the shader needs no
    /// special case for the common arrangement.
    #[test]
    fn colour_repeats_the_density_when_no_field_is_given() {
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = volume_texture(&values, None, &grid([2, 2, 2]), None).expect("built");
        let data = image.data.expect("kept the pixels");
        for pair in data.chunks_exact(2) {
            assert_eq!(pair[0], pair[1]);
        }
    }

    /// Density and colour scale independently: an explicit colour range must not
    /// change how solid the volume is.
    #[test]
    fn the_two_channels_scale_independently() {
        let density: Vec<f32> = (0..8).map(|v| v as f32).collect();
        // Runs the other way, so a channel mix-up is visible rather than subtle.
        let tint: Vec<f32> = (0..8).map(|v| 7.0 - v as f32).collect();

        let image = volume_texture(&density, Some(&tint), &grid([2, 2, 2]), None).expect("built");
        let data = image.data.expect("kept the pixels");
        assert_eq!((data[0], data[1]), (0, 255), "lowest density, highest tint");
        assert_eq!(
            (data[14], data[15]),
            (255, 0),
            "highest density, lowest tint"
        );
    }

    /// A colour field that does not cover the grid is dropped rather than
    /// fataled: the volume still has a shape worth seeing.
    #[test]
    fn a_short_colour_field_falls_back_to_the_density() {
        let density: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = volume_texture(&density, Some(&[0.0, 1.0]), &grid([2, 2, 2]), None)
            .expect("built anyway");
        let data = image.data.expect("kept the pixels");
        assert_eq!(data[0], data[1]);
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

        let image = volume_texture(&values, None, &grid([2, 2, 2]), None).expect("built");
        let data = image.data.expect("kept the pixels");
        // Texture order is x fastest, and two bytes to a sample, so the density
        // channel of neighbouring samples sits two apart.
        assert_eq!(data[0], 0, "x = 0");
        assert_eq!(data[2], 255, "x = 1");
        assert_eq!(data[4], 0, "x = 0 of the next row");
        assert_eq!(data[6], 255, "x = 1 of the next row");
    }

    /// An explicit colour range wins over the data's own, so two volumes can be
    /// compared against the same scale. It governs colour only — the density
    /// always scales against itself, or changing the scale would change how
    /// solid the volume looks.
    #[test]
    fn an_explicit_range_governs_colour_only() {
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let image = volume_texture(&values, Some(&values), &grid([2, 2, 2]), Some((0.0, 14.0)))
            .expect("built");
        let data = image.data.expect("kept the pixels");
        // Density: 7 of 7 is the top.
        assert_eq!(data[14], 255);
        // Colour: 7 of 14 is half way.
        assert!((120..=136).contains(&data[15]), "got {}", data[15]);
    }

    /// A density field that does not cover the grid is refused rather than
    /// sampled off the end of its own array.
    #[test]
    fn a_short_density_field_is_refused() {
        assert!(volume_texture(&[0.0, 1.0], None, &grid([4, 4, 4]), None).is_none());
    }
}
