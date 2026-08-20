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

use iris3d_model::{ParamKind, ParamSpec, float};
use iris3d_scene::registry::{ActorKind, ActorRegistry};
use iris3d_scene::{DataArray, DataStore, Dtype};

use iris3d_scene::link::Placement;

use super::{Actor, Dirty, mark};

/// What this pathway needs to redraw a point cloud.
///
/// Points are opaque, so they go through Bevy's ordinary passes and carry a
/// mesh and a material like any lit geometry — nothing here touches the moment
/// buffer.
type Drawable<'a> = (
    Actor<'a, PointsStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<PointQuadMaterial>>,
);

/// Points drawn as camera-facing discs. `size` is a diameter in world units, so
/// a sensible value depends on the data's own scale — there is no universally
/// right default.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PointsStyle {
    pub size: f32,
    /// Linear RGB, used where nothing is bound to `colour`.
    pub tint: Vec3,
}

const PARAMS: &[ParamSpec] = &[
    // What this needs to draw anything, stated rather than inferred from an
    // array happening to be called "positions". A client binds whatever it
    // uploaded, under whatever name.
    ParamSpec {
        id: "positions",
        label: "positions",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    // Linear RGB, one triple per point, already mapped. Unbound takes `tint`.
    //
    // This takes colours rather than the numbers to make colours *from*. What
    // ramp, over what range, is the `colormap` filter's business — so colouring
    // by anything a client can compute costs no change here.
    ParamSpec {
        id: "colour",
        label: "colour",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: false,
            structural: false,
        },
    },
    crate::TINT,
    ParamSpec {
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
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "points",
        label: "points",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(PointsStyle {
                size: float(params, "size", 0.05),
                tint: crate::tint(params, "tint", Vec3::splat(0.8)),
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
const SHADER: &str = "embedded://app/draw/default/point_quad.wgsl";

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
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, bound, dirty), mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        // The actor's own binding rather than its source object's dataset. The
        // object is a place in the tree now; the data is whatever was bound.
        let Some(array) = bound
            .get("positions")
            .and_then(|id| store.array(id))
            .and_then(|held| arrays.get(&held.handle))
        else {
            // `check_bindings` refuses an actor with no positions, so reaching
            // here means the array was released from under a living actor.
            continue;
        };

        let all = array.to_vec3();
        if all.is_empty() {
            continue;
        }
        let centres = all;
        let count = centres.len();

        if dirty.geometry || dirty.colour {
            let bound_rgb = super::bound(bound, "colour", &store, &arrays)
                .and_then(|values| crate::bound_colours(values, array.count() as usize));
            let flat = style.tint.extend(1.0).to_array();
            let colours = quad_colours(count, bound_rgb.as_ref(), flat);

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
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }

                let mut mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                );
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
                mesh.insert_indices(Indices::U32(indices));
                ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, mesh);
                debug!("draw: {count} point quads rebuilt");
            } else {
                repaint(&mut meshes, mesh3d, colours);
                debug!("draw: {count} point quads repainted");
            }
        }

        // After the geometry branch, which may have created the material this
        // then writes through.
        if dirty.material || dirty.geometry {
            ensure_material(
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

/// Replaces a mesh in place when the actor already has one, so a rebuild does
/// not leak a fresh  into  on every change.
fn ensure_mesh(
    commands: &mut Commands,
    entity: Entity,
    meshes: &mut Assets<Mesh>,
    existing: Option<&Mesh3d>,
    mesh: Mesh,
) {
    if let Some(Mesh3d(handle)) = existing
        && let Some(mut slot) = meshes.get_mut(handle)
    {
        *slot = mesh;
        return;
    }
    commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
}

/// As [`ensure_mesh`], for the material.
fn ensure_material(
    commands: &mut Commands,
    entity: Entity,
    materials: &mut Assets<PointQuadMaterial>,
    existing: Option<&MeshMaterial3d<PointQuadMaterial>>,
    material: PointQuadMaterial,
) {
    if let Some(MeshMaterial3d(handle)) = existing
        && let Some(mut slot) = materials.get_mut(handle)
    {
        *slot = material;
        return;
    }
    commands
        .entity(entity)
        .insert(MeshMaterial3d(materials.add(material)));
}

/// Overwrites vertex colours without touching anything else. Only legal when
/// the vertex count is unchanged, which is exactly when the geometry is not
/// also dirty.
fn repaint(meshes: &mut Assets<Mesh>, existing: Option<&Mesh3d>, colours: Vec<[f32; 4]>) {
    let Some(Mesh3d(handle)) = existing else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(handle) else {
        return;
    };
    if mesh.count_vertices() != colours.len() {
        warn!(
            "draw: {} vertex colours for a mesh of {} vertices; skipping the repaint",
            colours.len(),
            mesh.count_vertices()
        );
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

/// Gives every placement the mesh and material the actor holds.
#[allow(clippy::type_complexity)]
pub fn place_points(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MeshMaterial3d<PointQuadMaterial>), With<PointsStyle>>,
    placements: Query<(
        Entity,
        &Placement,
        Option<&Mesh3d>,
        Option<&MeshMaterial3d<PointQuadMaterial>>,
    )>,
) {
    for (entity, placement, mesh3d, material3d) in &placements {
        let Ok((mesh, material)) = actors.get(placement.0) else {
            continue;
        };
        if mesh3d.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert(mesh.clone());
        }
        if material3d.map(|MeshMaterial3d(handle)| handle.id()) != Some(material.0.id()) {
            commands.entity(entity).insert(material.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;
    use iris3d_data::array::{BufferMeta, Dtype};
    use iris3d_model::Bindings;
    use iris3d_scene::{ActorKindId, SceneObject};

    /// Runs the invalidation chain and this kind, with no renderer behind it:
    /// everything being asserted is about assets, not pixels.
    ///
    /// Chained by hand rather than through the plugin's sets, because the point
    /// is one kind in isolation — adding `DrawPlugin` would drag in every other
    /// kind and a `MaterialPlugin` besides. The shared halves come from
    /// `crate`, which is two levels up now that this kind belongs to a
    /// backend.
    fn app() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        // Geometry is an asset like any other, and `mark_dirty` watches it: a
        // filter rewriting a mesh has to reach the actors drawing it.
        app.add_message::<AssetEvent<Mesh>>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<DataStore>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<PointQuadMaterial>>();
        // `mark_dirty` asks the registry what a changed array invalidates, so
        // this kind has to be registered even though nothing here reads its
        // parameters — the real `register` is used, not a stand-in, so the
        // declarations under test are the ones that ship.
        app.init_resource::<iris3d_scene::ActorRegistry>();
        register(
            &mut app
                .world_mut()
                .resource_mut::<iris3d_scene::ActorRegistry>(),
        );
        app.add_systems(
            Update,
            (
                (crate::mark_dirty, invalidate),
                draw_points,
                crate::clear_dirty,
            )
                .chain(),
        );

        let meta = BufferMeta {
            name: "whatever".into(),
            dtype: Dtype::Float32,
            shape: vec![4, 3],
        };
        let positions = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(DataArray::numeric(Dtype::Float32, vec![4, 3], vec![0; 48]));
        // Held by handle, with no object involved: the array is not the object's
        // any more, and the name it arrived under carries no meaning.
        app.world_mut()
            .resource_mut::<DataStore>()
            .insert(0, meta, positions);
        // The object is only a place in the tree now.
        let object = app
            .world_mut()
            .spawn(SceneObject {
                name: "somewhere".into(),
            })
            .id();
        let actor = app
            .world_mut()
            .spawn((
                ActorKindId("points"),
                PointsStyle {
                    size: 0.05,
                    tint: Vec3::splat(0.8),
                },
                Bindings(HashMap::from_iter([("positions", 0)])),
            ))
            .id();

        app.update();
        (app, object, actor)
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
        let (app, _, actor) = app();
        assert_eq!(counts(&app), (1, 1));
        assert_eq!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get(mesh_of(&app, actor))
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
        let (mut app, _, actor) = app();
        let (mesh, material) = (mesh_of(&app, actor), material_of(&app, actor));

        // The flat colour is a parameter now, so this goes through the style
        // component the way any other setting does.
        app.world_mut().get_mut::<PointsStyle>(actor).unwrap().tint = Vec3::new(1.0, 0.0, 0.0);
        crate::mark(&mut app.world_mut().commands(), actor, crate::Dirty::COLOUR);
        app.update();

        assert_eq!(mesh_of(&app, actor), mesh, "mesh should be reused");
        assert_eq!(material_of(&app, actor), material);
        assert_eq!(counts(&app), (1, 1), "nothing should have been allocated");
    }

    /// Point size is a shader uniform, so changing it touches neither the mesh
    /// nor the material *asset* — only the value inside it.
    #[test]
    fn resizing_reuses_both_assets() {
        let (mut app, _, actor) = app();
        let (mesh, material) = (mesh_of(&app, actor), material_of(&app, actor));

        app.world_mut().get_mut::<PointsStyle>(actor).unwrap().size = 0.5;
        app.update();

        assert_eq!(mesh_of(&app, actor), mesh);
        assert_eq!(material_of(&app, actor), material);
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

    /// Fewer points in means fewer quads out, and the assets are reused rather
    /// than reallocated.
    ///
    /// The kind narrows nothing — a `gather` upstream does — so what it draws
    /// is simply what it is bound to, and rebinding is how the drawn set
    /// changes.
    #[test]
    fn rebinding_fewer_positions_draws_fewer_points() {
        let (mut app, _, actor) = app();
        let narrowed = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(DataArray::numeric(
                Dtype::Float32,
                vec![2, 3],
                [0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ));
        let handle = 77u64;
        app.world_mut().resource_mut::<DataStore>().insert(
            handle,
            iris3d_scene::BufferMeta {
                name: "positions".into(),
                dtype: Dtype::Float32,
                shape: vec![2, 3],
            },
            narrowed,
        );
        app.world_mut()
            .get_mut::<Bindings>(actor)
            .unwrap()
            .0
            .insert("positions", handle);
        app.update();

        assert_eq!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get(mesh_of(&app, actor))
                .map(|mesh| mesh.count_vertices()),
            Some(8),
            "two points, four vertices each"
        );
        assert_eq!(counts(&app), (1, 1), "still reusing the same assets");
    }

    /// A hundred slider frames should leave exactly the assets one frame does.
    #[test]
    fn dragging_a_slider_allocates_nothing() {
        let (mut app, _, actor) = app();
        for step in 0..100 {
            app.world_mut().get_mut::<PointsStyle>(actor).unwrap().size =
                0.01 + step as f32 * 0.001;
            app.update();
        }
        assert_eq!(counts(&app), (1, 1));
    }
}
