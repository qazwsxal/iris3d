//! Sampled grids, drawn as an emitting and absorbing medium.
//!
//! A scalar field on a regular grid is one physical thing whether it is ray
//! marched onto a standard pipeline or integrated into moments here. What a
//! pathway decides is the picture, not the object, which is why the id says
//! `volume` and nothing about the technique.
//!
//! # Two passes, because a volume both absorbs and emits
//!
//! `volume.wgsl` deposits the absorbing half into the shared moment buffer, so a
//! volume dims whatever lies behind it — including opaque geometry sitting
//! *inside* it, because the accumulation truncates every interval at the opaque
//! depth. That half is exact: the fullscreen resolve reads `b0`, and `b0` after
//! truncation is precisely the absorbance in front of the lit surface.
//!
//! `emit.wgsl` then marches the same ray again and adds what the volume gives
//! off, attenuated by everything in front of each step. It has to run second
//! because "everything in front" includes other volumes and meshes, which only
//! the completed moment buffer knows about.
//!
//! # What the parameters mean
//!
//! `density` is what makes the volume solid and `colour` is what tints it, and
//! they are deliberately separate bindings — density from one quantity and
//! colour from another is the usual pairing in scientific volume rendering. The
//! grid itself arrives as nine numbers rather than an array, because 64³ samples
//! state their arrangement far more cheaply than 262144 coordinates can.
//!
//! `opacity` and `emission` are separate controls on purpose. One decides how
//! much the volume blocks and the other how much it gives off; a volume that
//! glowed in proportion to its own opacity could not be made bright without also
//! being made opaque.
//!
//! # What is not here
//!
//! No empty-space skipping beyond the zero-density test in the shader, no
//! gradient lighting, and no transfer function worth the name — colour is a
//! fixed ramp and density scales linearly. All three are real improvements and
//! none belongs in a first pass.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::scene::link::Placement;
use crate::scene::registry::{
    ActorKind, ActorRegistry, Bindings, ParamKind, ParamSpec, float, text,
    uvec3 as param_uvec3, vec3 as param_vec3, vector,
};
use crate::filter::colormap::ColorMap;
use crate::scene::{DataArray, DataStore};

use super::{Actor, Dirty, bound, mark};

/// How the medium absorbs and emits.
///
/// Kept apart from [`GridBox`] for the reason the default backend keeps its two
/// apart: these are material properties that go to the shader as a uniform and
/// touch nothing else, while the grid decides the box and the texture upload.
/// One component for both would make an opacity drag re-upload a 64³ texture,
/// which is exactly what grading [`Dirty`] exists to avoid.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct GridStyle {
    /// Which ramp the `colour` channel is read through, and over what range.
    ///
    /// A transfer function rather than a colour array, and the reason `volume`
    /// still maps its own values while every other kind takes RGB from the
    /// `colormap` filter: this is evaluated per *sample along the ray*, millions
    /// of times a frame, not once per element. Materialising an RGB triple per
    /// voxel would cost 200 MB at 256³ to save a texture fetch.
    pub map: ColorMap,
    /// The medium's extinction colour as linear RGB, read as a transmission.
    pub tint: Vec3,
    /// Value range the map spans, or `None` to autoscale over the field.
    pub range: Option<(f32, f32)>,
    /// Absorbance per world unit at a density of 1.
    pub sigma: f32,
    /// Emitted radiance per world unit at a density of 1.
    pub emission: f32,
    /// Samples along the ray. A quality control: the step length divides out of
    /// the integral, so the picture holds still as this moves.
    pub steps: f32,
}

/// Where the samples sit.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct GridBox {
    pub origin: Vec3,
    pub spacing: Vec3,
    pub dims: UVec3,
}

impl GridBox {
    pub fn point_count(&self) -> usize {
        self.dims.x as usize * self.dims.y as usize * self.dims.z as usize
    }

    /// Extent of the box the samples span. An axis of one sample spans nothing,
    /// which is what a single slice uploaded as a grid looks like — hence the
    /// saturating subtraction rather than one that wraps.
    pub fn size(&self) -> Vec3 {
        self.dims.saturating_sub(UVec3::ONE).as_vec3() * self.spacing
    }

    /// Maps the unit cube `[0,1]³` onto this box, which is the space both
    /// shaders slab-test in and sample the texture with.
    pub fn cube_to_box(&self) -> Mat4 {
        // A degenerate axis would make the matrix singular, and the shaders need
        // its inverse. One sample thick is legitimate input, so it is nudged
        // rather than refused.
        let size = self.size().max(Vec3::splat(1.0e-6));
        Mat4::from_translation(self.origin) * Mat4::from_scale(size)
    }
}

/// The uploaded field, and the ramp the emission is coloured by.
///
/// The drawable output of this kind: what `extract` picks up and what a
/// placement is given a copy of.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct GridField {
    /// `Rgba8Unorm`: what absorbs in red, what the ramp is read by in green,
    /// what emits in blue. One tap serves all three, so a step costs one fetch
    /// however many separate arrays were bound.
    pub field: Handle<Image>,
    pub ramp: Handle<Image>,
    /// The medium's extinction colour, as a *transmission*. The `tint`
    /// parameter, so it means the same thing here as it does for a solid.
    pub tint: Vec3,
}

/// The pinned range, or `None` when the ends are equal and it autoscales.
///
/// Equal ends carry no information — every value would land at the same point —
/// so they are the natural spelling of "work it out from the field". Spelled the
/// same way as the `colormap` filter's range, so the two agree.
fn pinned(params: &crate::scene::registry::ParamMap) -> Option<(f32, f32)> {
    let range = vector(params, "range", 2);
    (range[0] < range[1]).then(|| (range[0] as f32, range[1] as f32))
}

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "density",
        label: "density",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    // What glows, which need not be what blocks. Density from one quantity and
    // emission from another is the usual pairing — a cloud whose *density*
    // absorbs while its *temperature* radiates — and tying the two together
    // makes that impossible to say. Unbound means glow in proportion to
    // density, which is the sensible default and what a single-field volume
    // wants.
    ParamSpec {
        id: "emissive",
        label: "emission from",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    // Unbound means colour by the density itself.
    //
    // `structural`, unlike the same input on `surface` and `points`, because there
    // are no vertices to repaint. These values are packed into the green channel
    // of the 3D field texture, so new ones mean a new texture — the whole
    // upload, not four bytes a vertex. The cheap path here is a *ramp* change,
    // which `Dirty::COLOUR` already covers and which touches 256 texels.
    ParamSpec {
        id: "colour",
        label: "colour by",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    // The transfer function. `volume` maps its own values, unlike every other
    // kind, because the ramp is read per sample along the ray rather than once
    // per element — see `GridStyle::map`.
    ParamSpec {
        id: "map",
        label: "colour map",
        kind: ParamKind::Choice {
            options: crate::filter::colormap::MAPS,
            default: "viridis",
        },
    },
    // Equal ends autoscale over the bound field, exactly as the `colormap`
    // filter's range does. Spelled the same way so the two agree.
    ParamSpec {
        id: "range",
        label: "range (equal ends autoscale)",
        kind: ParamKind::Vector {
            components: 2,
            default: &[0.0, 0.0],
            min: -1.0e30,
            max: 1.0e30,
            integral: false,
        },
    },
    crate::draw::TINT,
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
    // Logarithmic for the same reason the surface's `sigma` is: the interesting
    // part of the range is at the bottom of it.
    ParamSpec {
        id: "opacity",
        label: "absorbance per unit",
        kind: ParamKind::Float {
            default: 1.0,
            min: 0.001,
            max: 50.0,
            logarithmic: true,
        },
    },
    ParamSpec {
        id: "emission",
        label: "emission per unit",
        kind: ParamKind::Float {
            default: 1.0,
            min: 0.0,
            max: 50.0,
            logarithmic: true,
        },
    },
    ParamSpec {
        id: "steps",
        label: "steps",
        kind: ParamKind::Float {
            default: 128.0,
            min: 8.0,
            max: 1024.0,
            logarithmic: false,
        },
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "volume",
        // Plain on purpose. One volume both absorbs and emits — `opacity` and
        // `emission` are two controls on one medium, not two kinds of object —
        // so naming it after either half would describe it wrongly.
        label: "volume",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert((
                GridStyle {
                    tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
                    map: ColorMap::from_str(text(params, "map", "viridis")).unwrap_or_default(),
                    range: pinned(params),
                    sigma: float(params, "opacity", 1.0),
                    emission: float(params, "emission", 1.0),
                    steps: float(params, "steps", 128.0),
                },
                GridBox {
                    origin: param_vec3(params, "origin", Vec3::ZERO),
                    spacing: param_vec3(params, "spacing", Vec3::ONE),
                    dims: param_uvec3(params, "dims", UVec3::ONE),
                },
            ));
        },
    });
}

/// What a change to this kind's own parameters invalidates.
///
/// The split is the whole point of grading [`Dirty`]. A style change is a
/// uniform write — the shaders read `sigma`, `emission` and `steps` straight out
/// of the grid uniform — so it must not re-upload the texture. Changing the grid
/// does re-upload it, because the texture's very dimensions come from there.
pub fn invalidate(
    mut commands: Commands,
    restyled: Query<Entity, (With<GridStyle>, Changed<GridStyle>)>,
    regridded: Query<Entity, (With<GridStyle>, Changed<GridBox>)>,
) {
    for entity in &restyled {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
    for entity in &regridded {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

/// Builds each dirty volume's field texture and colour ramp.
#[allow(clippy::too_many_arguments)]
pub fn draw_volumes(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    actors: Query<(Actor<GridStyle>, &GridBox, Option<&GridField>)>,
) {
    for ((entity, style, _subset, bindings, dirty), grid, existing) in &actors {
        // Colour-only changes repaint the ramp and leave the field alone; a
        // material change touches neither, because both live in the uniform.
        if !dirty.geometry && !dirty.colour {
            continue;
        }

        let ramp = images.add(super::super::ramp_texture(style.map));

        // A repaint keeps the field it already uploaded. This is the difference
        // between dragging the colour map on a 128³ volume costing a 256-texel
        // ramp and costing a four-megabyte re-upload.
        let reusable = existing.filter(|_| !dirty.geometry);
        let field = match reusable {
            Some(existing) => existing.field.clone(),
            None => {
                let Some(image) = field_texture(grid, bindings, &store, &arrays, style) else {
                    continue;
                };
                images.add(image)
            }
        };

        commands.entity(entity).insert(GridField {
            field,
            ramp,
            // Read as a transmission, exactly as a solid's tint is: this is what
            // the medium lets through, not what it looks like.
            tint: style.tint,
        });
    }
}

/// Packs the bound arrays into one `Rgba8Unorm` 3D texture.
///
/// Three quantities, one fetch: `r` is what absorbs, `g` is what the ramp is
/// read by, `b` is what emits. Keeping them in one texture is what lets a step
/// cost a single tap however many of them are bound separately — and each is
/// normalised over its own range, so the ends of a channel are the extremes of
/// that quantity rather than of some other one.
///
/// Eight bits per channel is what buys hardware filtering, which is the whole
/// mechanism behind "nearest neighbour versus linear is a sampler setting" —
/// see the header of `volume.wgsl`.
///
/// An unbound optional falls back to the density, so a volume that names one
/// field still glows and colours by it. That is why the shaders need no flag
/// saying which channel to read: the fallback happens here, once, rather than
/// as a branch per sample.
fn field_texture(
    grid: &GridBox,
    bindings: &Bindings,
    store: &DataStore,
    arrays: &Assets<DataArray>,
    style: &GridStyle,
) -> Option<Image> {
    let expected = grid.point_count();
    if expected == 0 {
        return None;
    }

    let density = bound(bindings, "density", store, arrays)?.to_f32();
    if density.len() < expected {
        warn!(
            "draw: a volume's density field has {} values for a grid of {expected}",
            density.len()
        );
        return None;
    }

    // A field that does not cover the grid is dropped rather than fataled: the
    // volume still has a shape worth seeing.
    let optional = |id: &str| {
        bound(bindings, id, store, arrays)
            .map(|array| array.to_f32())
            .filter(|values| {
                let long_enough = values.len() >= expected;
                if !long_enough {
                    warn!(
                        "draw: a volume's {id} field is shorter than its grid; \
                         falling back to the density"
                    );
                }
                long_enough
            })
    };
    let tint_values = optional("colour");
    let glow_values = optional("emissive");

    let density_range = range_of(&density[..expected]);
    // Only the colour ramp honours an explicit `range` — that control says where
    // the *map* starts and ends, and applying it to emission would silently
    // rescale brightness when someone pinned a colour scale.
    let colour_range = tint_values
        .as_ref()
        .map(|values| style.range.unwrap_or_else(|| range_of(&values[..expected])));
    let glow_range = glow_values
        .as_ref()
        .map(|values| range_of(&values[..expected]));

    // Reorder while normalising. The wire runs z fastest, which is what a numpy
    // array of shape (x, y, z) gives from a plain `.ravel()`. A 3D texture wants
    // x fastest. Getting this wrong does not fail — it silently transposes the
    // volume, which looks like a plausible render of the wrong thing.
    let (nx, ny, nz) = (
        grid.dims.x as usize,
        grid.dims.y as usize,
        grid.dims.z as usize,
    );
    let channel = |values: &Option<Vec<f32>>, range: Option<(f32, f32)>, index: usize, fallback| {
        match (values, range) {
            (Some(values), Some(range)) => quantise(values[index], range),
            _ => fallback,
        }
    };
    let mut data = Vec::with_capacity(expected * 4);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let index = (x * ny + y) * nz + z;
                let red = quantise(density[index], density_range);
                data.push(red);
                data.push(channel(&tint_values, colour_range, index, red));
                data.push(channel(&glow_values, glow_range, index, red));
                data.push(255);
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
        TextureFormat::Rgba8Unorm,
        // The CPU copy is dead weight once uploaded, and this is rebuilt from
        // the `DataArray` whenever it changes.
        RenderAssetUsages::RENDER_WORLD,
    );
    // Linear filtering is the whole reason for the 8-bit format. Swapping this
    // for `ImageSampler::nearest()` turns the volume into per-cell slabs and
    // changes nothing else.
    image.sampler = ImageSampler::linear();
    Some(image)
}

/// The extremes of a field, ignoring anything that is not finite.
fn range_of(values: &[f32]) -> (f32, f32) {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for value in values {
        if value.is_finite() {
            low = low.min(*value);
            high = high.max(*value);
        }
    }
    if low > high { (0.0, 1.0) } else { (low, high) }
}

fn quantise(value: f32, (low, high): (f32, f32)) -> u8 {
    let span = if (high - low).abs() < f32::EPSILON {
        1.0
    } else {
        high - low
    };
    (((value - low) / span).clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Gives every placement of an actor the field and the style that actor owns.
///
/// The counterpart of `place_volumes` for grids, and separate from it because
/// the two kinds carry different components: `extract_grids` queries
/// [`GridField`], [`GridStyle`] and [`GridBox`] together, so a placement needs
/// all three or it is not a volume at all.
#[allow(clippy::type_complexity)]
pub fn place_grids(
    mut commands: Commands,
    actors: Query<(&GridField, &GridStyle, &GridBox)>,
    placements: Query<(
        Entity,
        &Placement,
        Option<&GridField>,
        Option<&GridStyle>,
        Option<&GridBox>,
    )>,
) {
    for (entity, placement, current_field, current_style, current_box) in &placements {
        let Ok((field, style, grid)) = actors.get(placement.0) else {
            continue;
        };
        // All three compared before writing, so a settled scene inserts nothing
        // and `Changed` filters downstream stay quiet.
        if current_field != Some(field) {
            commands.entity(entity).insert(field.clone());
        }
        if current_style != Some(style) {
            commands.entity(entity).insert(*style);
        }
        if current_box != Some(grid) {
            commands.entity(entity).insert(*grid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_slice_still_has_an_invertible_box() {
        let grid = GridBox {
            origin: Vec3::ZERO,
            spacing: Vec3::ONE,
            dims: UVec3::new(4, 4, 1),
        };
        // The z axis spans nothing, which is what a slice looks like. The
        // shaders need the inverse of this matrix, so a singular one would take
        // the whole volume off screen rather than flattening it.
        let inverse = grid.cube_to_box().inverse();
        assert!(inverse.is_finite(), "got {inverse:?}");
    }

    #[test]
    fn the_box_spans_the_samples_not_the_cells() {
        let grid = GridBox {
            origin: Vec3::ZERO,
            spacing: Vec3::splat(2.0),
            dims: UVec3::splat(5),
        };
        // Five samples two apart span eight units, not ten: the extent is
        // between the first and last sample, not around them.
        assert_eq!(grid.size(), Vec3::splat(8.0));
    }

    /// The wire runs z fastest and a 3D texture wants x fastest. A transpose
    /// here renders a plausible picture of the wrong thing, so it is worth
    /// pinning rather than trusting.
    #[test]
    fn the_upload_reorders_z_fastest_into_x_fastest() {
        let (nx, ny, nz) = (2usize, 1usize, 3usize);
        // Value equals the wire index, so the reorder is visible in the output.
        let wire: Vec<f32> = (0..(nx * ny * nz)).map(|index| index as f32).collect();

        let mut got = Vec::new();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    got.push(wire[(x * ny + y) * nz + z]);
                }
            }
        }
        // x fastest: (x=0,z=0), (x=1,z=0), (x=0,z=1), ...
        assert_eq!(got, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn quantising_puts_the_extremes_at_the_ends() {
        let range = (-2.0, 6.0);
        assert_eq!(quantise(-2.0, range), 0);
        assert_eq!(quantise(6.0, range), 255);
        assert_eq!(quantise(-99.0, range), 0, "clamped, not wrapped");
    }

    /// A constant field has no range at all, and dividing by its width would
    /// produce a texture of NaN.
    #[test]
    fn a_flat_field_does_not_divide_by_zero() {
        assert_eq!(quantise(3.0, (3.0, 3.0)), 0);
    }

    #[test]
    fn a_field_of_nothing_but_nans_falls_back_to_a_unit_range() {
        assert_eq!(range_of(&[f32::NAN, f32::INFINITY]), (0.0, 1.0));
    }
}
