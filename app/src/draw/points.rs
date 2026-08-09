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
use crate::scene::{DataArray, DatasetKind, PointCloud};

use super::{Dirty, Drawable, mark};

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

/// `size` reaches the shader as a uniform, so changing it is a material write
/// and never touches the mesh.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<PointsStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
}

/// One vertex colour per corner, so a repaint writes the same value four times.
fn quad_colours(centres: usize, tint: Option<&Vec<[f32; 4]>>, flat: [f32; 4]) -> Vec<[f32; 4]> {
    let mut colours = Vec::with_capacity(centres * 4);
    for index in 0..centres {
        let rgba = tint.map_or(flat, |colours| colours[index]);
        colours.extend_from_slice(&[rgba; 4]);
    }
    colours
}

pub fn draw_points(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PointQuadMaterial>>,
    arrays: Res<Assets<DataArray>>,
    dirty: Query<Drawable<PointsStyle, PointQuadMaterial>>,
    clouds: Query<(&PointCloud, Option<&Fields>)>,
) {
    for (entity, style, colour, subset, source, dirty, mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Ok((cloud, fields)) = clouds.get(source.0) else {
            continue;
        };
        let Some(array) = arrays.get(&cloud.positions) else {
            continue;
        };

        let all = array.to_vec3();
        if all.is_empty() {
            continue;
        }
        // Points have no connectivity, so a subset is a plain filter — nothing
        // refers to a point by index, so nothing needs renumbering.
        let kept = subset.selected(all.len(), &arrays);
        let centres: Vec<Vec3> = match &kept {
            Some(kept) => kept.iter().map(|index| all[*index as usize]).collect(),
            None => all,
        };
        let count = centres.len();

        if dirty.geometry || dirty.colour {
            let tint = super::colour_field(colour, fields)
                .and_then(|field| super::vertex_colours(field, colour, &arrays, array.count() as usize))
                // Colours are computed over the whole field, then narrowed to
                // the drawn points, so a subset does not shift the mapping.
                .map(|colours| match &kept {
                    Some(kept) => kept.iter().map(|index| colours[*index as usize]).collect(),
                    None => colours,
                });
            let flat = colour.flat.to_linear().to_f32_array();
            let colours = quad_colours(count, tint.as_ref(), flat);

            if dirty.geometry {
                let mut positions = Vec::with_capacity(count * 4);
                let mut uvs = Vec::with_capacity(count * 4);
                let mut indices = Vec::with_capacity(count * 6);

                for (index, centre) in centres.iter().enumerate() {
                    let base = (index * 4) as u32;
                    for corner in CORNERS {
                        positions.push([centre.x, centre.y, centre.z]);
                        uvs.push(corner);
                    }
                    indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }

                let mut mesh =
                    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
                mesh.insert_indices(Indices::U32(indices));
                super::ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, mesh);
                debug!("draw: {count} point quads rebuilt");
            } else {
                super::repaint(&mut meshes, mesh3d, colours);
                debug!("draw: {count} point quads repainted");
            }
        }

        // After the geometry branch, which may have created the material this
        // then writes through.
        if dirty.material || dirty.geometry {
            super::ensure_material(
                &mut commands,
                entity,
                &mut materials,
                material3d,
                PointQuadMaterial {
                    params: Vec4::new(style.size, 0.0, 0.0, 0.0),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::data::{BufferMeta, Dtype, NamedArray};
    use crate::scene::{ColorBy, RepresentationKindId, RepresentationOf, SceneObject, Subset};

    /// Runs the invalidation chain and this backend, with no renderer behind
    /// it: everything being asserted is about assets, not pixels.
    fn app() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<PointQuadMaterial>>();
        app.add_systems(
            Update,
            (
                (super::super::mark_dirty, invalidate),
                draw_points,
                super::super::clear_dirty,
            )
                .chain(),
        );

        let positions = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(DataArray {
                dtype: Dtype::Float32,
                shape: vec![4, 3],
                data: vec![0; 48],
            });
        let object = app
            .world_mut()
            .spawn((
                SceneObject {
                    name: "cloud".into(),
                    arrays: vec![NamedArray {
                        meta: BufferMeta {
                            name: "positions".into(),
                            dtype: Dtype::Float32,
                            shape: vec![4, 3],
                        },
                        handle: positions.clone(),
                    }],
                },
                PointCloud { positions },
            ))
            .id();
        let representation = app
            .world_mut()
            .spawn((
                RepresentationKindId("points"),
                PointsStyle { size: 0.05 },
                ColorBy::default(),
                Subset::All,
                RepresentationOf(object),
            ))
            .id();

        app.update();
        (app, object, representation)
    }

    fn mesh_of(app: &App, entity: Entity) -> AssetId<Mesh> {
        app.world().get::<Mesh3d>(entity).expect("drawn").0.id()
    }

    fn material_of(app: &App, entity: Entity) -> AssetId<PointQuadMaterial> {
        app.world()
            .get::<MeshMaterial3d<PointQuadMaterial>>(entity)
            .expect("drawn")
            .0
            .id()
    }

    fn counts(app: &App) -> (usize, usize) {
        (
            app.world().resource::<Assets<Mesh>>().len(),
            app.world().resource::<Assets<PointQuadMaterial>>().len(),
        )
    }

    #[test]
    fn drawing_produces_one_mesh_and_one_material() {
        let (app, _, representation) = app();
        assert_eq!(counts(&app), (1, 1));
        assert_eq!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get(mesh_of(&app, representation))
                .map(|mesh| mesh.count_vertices()),
            Some(16),
            "four points, four vertices each"
        );
    }

    /// Recolouring rewrites the existing buffer. Before graded invalidation
    /// this allocated a whole new mesh, so dragging a colour-map slider leaked
    /// one per frame.
    #[test]
    fn recolouring_reuses_the_mesh() {
        let (mut app, _, representation) = app();
        let (mesh, material) = (mesh_of(&app, representation), material_of(&app, representation));

        app.world_mut()
            .get_mut::<ColorBy>(representation)
            .unwrap()
            .flat = Color::srgb(1.0, 0.0, 0.0);
        app.update();

        assert_eq!(mesh_of(&app, representation), mesh, "mesh should be reused");
        assert_eq!(material_of(&app, representation), material);
        assert_eq!(counts(&app), (1, 1), "nothing should have been allocated");
    }

    /// Point size is a shader uniform, so changing it touches neither the mesh
    /// nor the material *asset* — only the value inside it.
    #[test]
    fn resizing_reuses_both_assets() {
        let (mut app, _, representation) = app();
        let (mesh, material) = (mesh_of(&app, representation), material_of(&app, representation));

        app.world_mut()
            .get_mut::<PointsStyle>(representation)
            .unwrap()
            .size = 0.5;
        app.update();

        assert_eq!(mesh_of(&app, representation), mesh);
        assert_eq!(material_of(&app, representation), material);
        assert_eq!(counts(&app), (1, 1));
        assert_eq!(
            app.world()
                .resource::<Assets<PointQuadMaterial>>()
                .get(material)
                .map(|m| m.params.x),
            Some(0.5),
            "the new size should have reached the uniform"
        );
    }

    /// A subset reaches the vertex buffer, and changing it rebuilds rather than
    /// repaints — the vertex count moves, so a repaint would be wrong.
    #[test]
    fn a_subset_draws_fewer_points() {
        let (mut app, _, representation) = app();
        let indices = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(DataArray {
                dtype: Dtype::Uint32,
                shape: vec![2],
                data: [0u32, 2].iter().flat_map(|v| v.to_le_bytes()).collect(),
            });

        *app.world_mut().get_mut::<Subset>(representation).unwrap() = Subset::Selected {
            array: indices,
            encoding: crate::scene::SubsetEncoding::Indices,
            association: crate::scene::data::Association::PerPoint,
        };
        app.update();

        assert_eq!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get(mesh_of(&app, representation))
                .map(|mesh| mesh.count_vertices()),
            Some(8),
            "two of four points, four vertices each"
        );
        assert_eq!(counts(&app), (1, 1), "still reusing the same assets");
    }

    /// A hundred slider frames should leave exactly the assets one frame does.
    #[test]
    fn dragging_a_slider_allocates_nothing() {
        let (mut app, _, representation) = app();
        for step in 0..100 {
            app.world_mut()
                .get_mut::<PointsStyle>(representation)
                .unwrap()
                .size = 0.01 + step as f32 * 0.001;
            app.update();
        }
        assert_eq!(counts(&app), (1, 1));
    }
}
