//! Point clouds, drawn as camera-facing quads.
//!
//! Each point expands to four vertices sharing a centre, with the corner offset
//! in UV; the vertex shader displaces them in view space so they always face
//! the camera. That costs four vertices and two triangles per point, but unlike
//! `PointList` it honours [`PointsStyle::size`] and gives round points instead
//! of single pixels.
//!
//! At 250k points that is 1M vertices in one mesh — fine, but the point at
//! which real GPU instancing (one vertex buffer, one instance per point) starts
//! to pay for itself.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use crate::scene::data::Fields;
use crate::scene::registry::{
    float, ParamKind, ParamSpec, RepresentationKind, RepresentationRegistry,
};
use crate::scene::{ColorBy, DataArray, DatasetKind, PointCloud, RepresentationOf};

use super::NeedsRedraw;

/// Points drawn as camera-facing discs. `size` is a diameter in world units, so
/// a sensible value depends on the data's own scale — there is no universally
/// right default.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PointsStyle {
    pub size: f32,
}

const PARAMS: &[ParamSpec] = &[ParamSpec {
    id: "size",
    label: "size",
    kind: ParamKind::Float {
        default: 0.05,
        min: 0.001,
        max: 1.0,
        // Useful sizes span three orders of magnitude, so a linear slider
        // spends most of its travel on values nobody wants.
        logarithmic: true,
    },
}];

pub fn register(registry: &mut RepresentationRegistry) {
    registry.register(RepresentationKind {
        id: "points",
        label: "points",
        supports: |dataset| dataset == DatasetKind::Points,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(PointsStyle {
                size: float(params, "size", 0.05),
            });
        },
    });
}

/// The shader is compiled into the binary rather than loaded from an `assets`
/// directory.
///
/// Bevy resolves a filesystem asset root from `BEVY_ASSET_ROOT`, then
/// `CARGO_MANIFEST_DIR`, then — falling back — the *executable's* directory,
/// not the working directory. So `cargo run` finds `app/assets` via
/// `CARGO_MANIFEST_DIR` while launching `target/debug/app.exe` from a debugger
/// silently does not, and setting `cwd` does nothing to fix it. Point clouds
/// would then be the only thing missing, because everything else uses built-in
/// materials. Embedding removes the failure mode and the runtime file
/// dependency along with it.
const SHADER: &str = "embedded://app/draw/point_quad.wgsl";

/// Corner offsets in UV space, running -0.5..0.5 so the inscribed disc has
/// radius 0.5 and the quad's width equals the requested size.
const CORNERS: [[f32; 2]; 4] = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct PointQuadMaterial {
    /// `x` is the quad diameter in world units; the rest is padding.
    #[uniform(0)]
    pub params: Vec4,
}

impl Material for PointQuadMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }
}

pub fn draw_points(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PointQuadMaterial>>,
    arrays: Res<Assets<DataArray>>,
    dirty: Query<(Entity, &PointsStyle, &ColorBy, &RepresentationOf), With<NeedsRedraw>>,
    clouds: Query<(&PointCloud, Option<&Fields>)>,
) {
    for (entity, style, colour, source) in &dirty {
        let size = &style.size;
        let Ok((cloud, fields)) = clouds.get(source.0) else {
            continue;
        };
        let Some(array) = arrays.get(&cloud.positions) else {
            continue;
        };

        let centres = array.to_vec3();
        if centres.is_empty() {
            continue;
        }

        let tint = super::colour_field(colour, fields)
            .and_then(|field| super::vertex_colours(field, colour, &arrays, centres.len()));
        let flat = colour.flat.to_linear().to_f32_array();

        let count = centres.len();
        let mut positions = Vec::with_capacity(count * 4);
        let mut uvs = Vec::with_capacity(count * 4);
        let mut colours = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);

        for (index, centre) in centres.iter().enumerate() {
            let rgba = tint.as_ref().map_or(flat, |colours| colours[index]);
            let base = (index * 4) as u32;
            for corner in CORNERS {
                positions.push([centre.x, centre.y, centre.z]);
                uvs.push(corner);
                colours.push(rgba);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
        mesh.insert_indices(Indices::U32(indices));

        commands.entity(entity).insert((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(PointQuadMaterial {
                params: Vec4::new(*size, 0.0, 0.0, 0.0),
            })),
        ));

        debug!("draw: {count} point quads at size {size}");
    }
}
