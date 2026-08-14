//! The solari backend: raytraced lighting on `bevy_solari`.
//!
//! Two facts about Solari decide everything in this module.
//!
//! **It reads `StandardMaterial` and nothing else.** `bevy_solari`'s scene
//! binder builds its material table from `base_color`, the textures, `metallic`,
//! `perceptual_roughness`, `emissive` and `reflectance`. It never looks at
//! `Mesh::ATTRIBUTE_COLOR`. iris3d colours everything by vertex colour over a
//! white base, so a direct port of the default backend would draw every actor
//! flat white. Colour therefore has to arrive through the material, which is
//! what [`RampPalette`] and the ramp texture are for.
//!
//! **A raytraced mesh is instanced, not merged.** One mesh becomes one
//! acceleration structure, and every entity holding that handle is an instance
//! of it. Drawing a 250k-point cloud as one merged mesh of tessellated spheres
//! would be twenty million triangles; as instances it is one sphere's worth of
//! geometry and 250k transforms. That is the whole reason to prefer it, and it
//! forces the second colouring route: instances share their mesh's vertices, so
//! a per-instance colour cannot come from a vertex attribute.
//!
//! Solari also constrains the meshes themselves — see [`condition`]. Getting
//! that wrong fails inside Solari at runtime, not at compile time.
//!
//! What it cannot do at all: transparency, alpha masking, volumetrics, skinned
//! and morphed meshes. There is no `volume` kind here, deliberately. In
//! practice it wants an NVIDIA card, because it leans on DLSS ray
//! reconstruction.

use bevy::camera::{CameraMainTextureUsages, Hdr};
use bevy::mesh::Indices;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;
use bevy::solari::prelude::{RaytracingMesh3d, SolariLighting, SolariPlugins};

use crate::scene::actor::ColorMap;
use crate::scene::link::{Placement, Placements};
use crate::scene::registry::ActorRegistry;

pub(crate) use super::{
    Actor, Dirty, Draw, Invalidate, Place, RAMP_STEPS, bound, mark, normalised, ramp_texture,
};

mod molecule;
mod points;
mod mesh;

pub struct RtBackendPlugin;

impl Plugin for RtBackendPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SolariPlugins);

        {
            let mut registry = app.world_mut().resource_mut::<ActorRegistry>();
            points::register(&mut registry);
            mesh::register(&mut registry);
            molecule::register(&mut registry);
            // No `volume`. Solari has no volumetrics, so the kind is absent
            // rather than broken, and asking for it is refused by name.
        }

        app.init_resource::<RampPalette>()
            .init_resource::<Primitives>()
            .init_resource::<Accumulation>()
            .add_systems(
            Update,
            (
                (
                    points::invalidate,
                    mesh::invalidate,
                    molecule::invalidate,
                )
                    .in_set(Invalidate),
                (
                    points::draw_points,
                    mesh::draw_meshes,
                    molecule::draw_molecules,
                )
                    .in_set(Draw),
                    (place_instances, copy_surfaces).in_set(Place),
                ),
            );
        app.add_systems(Update, (raytrace_camera, match_view_targets, accumulate));
    }
}

/// How long to keep drawing after a change so the image can converge.
///
/// Raytraced lighting is a stochastic estimate refined over successive frames.
/// iris3d's window is deliberately reactive — it sleeps until something happens
/// (see [`crate::redraw`]) — because nothing in the scene moves on its own and
/// re-rendering an unchanged picture wastes a GPU that is often busy computing
/// the data being looked at. Those two facts are in direct conflict: left
/// alone, Solari gets a couple of frames per interaction and never resolves out
/// of its initial noise.
///
/// So this pathway buys frames back, but only the ones it needs: after anything
/// that changes the picture, ask for frames until the estimate has settled,
/// then let the loop go back to sleep. Converge, then idle — rather than
/// rendering forever.
#[derive(Resource)]
pub struct Accumulation {
    /// Frames to spend converging after a change. Zero disables it, which
    /// leaves the reactive behaviour untouched and the image noisy.
    pub frames: u32,
}

impl Default for Accumulation {
    fn default() -> Self {
        // Enough to resolve the worst of the noise without spinning the GPU for
        // an appreciable time after every slider nudge.
        Self { frames: 180 }
    }
}

/// Holds the redraw loop awake while the raytraced image is still converging.
///
/// The trigger is deliberately broad — any transform that moved, any actor
/// marked dirty — because *anything* visible changing invalidates the temporal
/// history. Judging it more finely would risk leaving a stale, noisy frame on
/// screen, which is the failure this exists to prevent.
fn accumulate(
    mut awake: ResMut<crate::redraw::KeepAwake>,
    settings: Res<Accumulation>,
    moved: Query<(), Changed<GlobalTransform>>,
    dirty: Query<&Dirty>,
    mut remaining: Local<u32>,
) {
    if !moved.is_empty() || dirty.iter().any(|dirty| dirty.any()) {
        *remaining = settings.frames;
    }
    if *remaining > 0 {
        awake.nudge();
        *remaining -= 1;
    }
}

/// The meshes every instanced kind draws copies of.
///
/// Built once, at unit size, and scaled per instance by the transform. One
/// sphere asset means one acceleration structure however many atoms or points
/// reference it, which is the entire economy of instancing here.
///
/// `ico(2)` is 320 triangles — enough to read as round at the sizes scientific
/// data is drawn at, and small enough that a quarter of a million copies is
/// still one small mesh on the GPU.
#[derive(Resource)]
pub(crate) struct Primitives {
    pub sphere: Handle<Mesh>,
    pub cylinder: Handle<Mesh>,
}

impl FromWorld for Primitives {
    fn from_world(world: &mut World) -> Self {
        let mut sphere = Sphere::new(1.0)
            .mesh()
            .ico(2)
            .expect("2 is a valid icosphere subdivision count");
        condition(&mut sphere);
        // Unit height and unit radius, along +Y, centred on the origin. A bond
        // scales and rotates this rather than building its own geometry.
        let mut cylinder = Cylinder::new(1.0, 1.0).mesh().resolution(12).build();
        condition(&mut cylinder);

        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        Self {
            sphere: meshes.add(sphere),
            cylinder: meshes.add(cylinder),
        }
    }
}

/// An actor's own single-colour material, for when nothing is bound to colour.
///
/// Held so a rebuild writes through the handle it already has. Without it every
/// redraw would leak a fresh `StandardMaterial` into `Assets`, which a slider
/// drag turns into hundreds.
#[derive(Component)]
pub(crate) struct FlatMaterial(pub Handle<StandardMaterial>);

/// Reuses an actor's flat material, creating it only the first time.
pub(crate) fn ensure_flat(
    commands: &mut Commands,
    entity: Entity,
    materials: &mut Assets<StandardMaterial>,
    existing: Option<&FlatMaterial>,
    colour: LinearRgba,
) -> Handle<StandardMaterial> {
    if let Some(FlatMaterial(handle)) = existing
        && let Some(mut slot) = materials.get_mut(handle)
    {
        *slot = flat(colour);
        return handle.clone();
    }
    let handle = materials.add(flat(colour));
    commands.entity(entity).insert(FlatMaterial(handle.clone()));
    handle
}

/// Gives the viewport's camera what Solari needs of it.
///
/// The viewport owns the camera and its navigation; a backend adds its own
/// requirements to whatever camera it finds. All three are hard requirements,
/// not preferences: Solari writes through a storage binding and does not work
/// with multisampling.
/// Puts every camera on the same view target description, not just the
/// raytraced one.
///
/// **This is what makes the backend visible at all**, so do not drop it as
/// tidying. A camera's view target is selected by its sample count (`Msaa`) and
/// by whether it carries [`Hdr`], which switches it onto an intermediate
/// high-dynamic-range texture. Two cameras drawing to one window that disagree
/// about either are writing to *different textures*.
///
/// iris3d draws the interface on a second camera layered over the first with
/// `ClearColorConfig::None` — see [`crate::ui`]. Solari requires `Msaa::Off` and
/// lights into an HDR target, so without this the UI camera composited onto a
/// texture the 3D pass had never touched: a working interface over a black
/// viewport, with the gizmos gone too, looking for all the world like the
/// backend had drawn nothing.
fn match_view_targets(mut commands: Commands, cameras: Query<Entity, Added<Camera>>) {
    for camera in &cameras {
        commands.entity(camera).insert((Msaa::Off, Hdr));
    }
}

fn raytrace_camera(
    mut commands: Commands,
    cameras: Query<(Entity, Option<&CameraMainTextureUsages>), Added<Camera3d>>,
) {
    for (camera, existing) in &cameras {
        // Added to whatever the camera already asked for, never replacing it.
        // Building this from `default()` discards usages something else put
        // there — and egui composites through the same texture, so dropping its
        // usage left the UI drawing over a 3D view that never reached the
        // screen.
        let usages = existing
            .copied()
            .unwrap_or_default()
            .with(TextureUsages::STORAGE_BINDING);
        commands
            .entity(camera)
            .insert((SolariLighting::default(), Msaa::Off, usages));
    }
}

/// Makes a mesh acceptable to Solari's acceleration-structure builder.
///
/// Four requirements, none of them checked at compile time and all of them
/// fatal at runtime: UVs must exist, tangents must exist, there must be no
/// second UV set, and indices must be 32-bit. Every mesh this backend produces
/// goes through here on its way out.
///
/// Tangents are generated last because `generate_tangents` needs the UVs and
/// the normals to already be there. It fails on a mesh with no indices or the
/// wrong topology, which is why the result is reported rather than unwrapped.
pub(crate) fn condition(mesh: &mut Mesh) {
    if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
        let vertices = mesh.count_vertices();
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.5]; vertices]);
    }
    // A second UV set is not merely unused: Solari rejects it.
    if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
    }
    if let Some(Indices::U16(narrow)) = mesh.indices() {
        let widened: Vec<u32> = narrow.iter().map(|index| *index as u32).collect();
        mesh.insert_indices(Indices::U32(widened));
    }
    if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT)
        && let Err(err) = mesh.generate_tangents()
    {
        warn!("draw: solari could not generate tangents: {err}");
    }
}

/// Writes each element's place in the colour map into `UV.x`.
///
/// The per-vertex half of the colouring story, for a mesh whose scalar varies
/// across it. `UV.y` is the middle of the ramp image, which is one texel tall.
/// `per_vertex` maps a vertex index to the element it belongs to, because one
/// element can own several vertices.
pub(crate) fn ramp_uvs(mesh: &mut Mesh, ramp: &[f32], per_vertex: impl Fn(usize) -> usize) {
    let count = mesh.count_vertices();
    let uvs: Vec<[f32; 2]> = (0..count)
        .map(|vertex| [ramp.get(per_vertex(vertex)).copied().unwrap_or(0.0), 0.5])
        .collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
}

/// Materials standing in for the colour map, one per step.
///
/// The per-instance half of the colouring story. Instances of one mesh share
/// its vertices, so a per-instance colour cannot be a vertex attribute and has
/// to be the material. Quantising to [`RAMP_STEPS`] makes that finite: a
/// quarter of a million points reference at most 256 materials, and 256 is the
/// resolution the ramp texture has anyway, so the two routes to a colour agree
/// to within a step.
///
/// Built once per map and kept for the life of the app. Four maps of 256 tiny
/// materials is a rounding error, and rebuilding them per frame would not be.
#[derive(Resource, Default)]
pub(crate) struct RampPalette {
    steps: HashMap<ColorMap, Vec<Handle<StandardMaterial>>>,
    /// The same maps as 1D images, for the per-vertex route. Cached for the
    /// same reason as the materials: a colour map does not change when a
    /// slider moves, so neither should the texture.
    images: HashMap<ColorMap, Handle<Image>>,
}

impl RampPalette {
    /// The colour map as a texture, for a mesh whose scalar varies across it.
    pub(crate) fn image(&mut self, map: ColorMap, images: &mut Assets<Image>) -> Handle<Image> {
        self.images
            .entry(map)
            .or_insert_with(|| images.add(ramp_texture(map)))
            .clone()
    }
}

impl RampPalette {
    /// The material for a normalised value, building the map's palette if this
    /// is the first time it has been asked for.
    pub(crate) fn pick(
        &mut self,
        map: ColorMap,
        materials: &mut Assets<StandardMaterial>,
        t: f32,
    ) -> Handle<StandardMaterial> {
        let palette = self.steps.entry(map).or_insert_with(|| {
            (0..RAMP_STEPS)
                .map(|step| {
                    let rgba = super::sample(map, step as f32 / (RAMP_STEPS - 1) as f32);
                    materials.add(flat(LinearRgba::from_f32_array(rgba)))
                })
                .collect()
        });
        let step = (t.clamp(0.0, 1.0) * (RAMP_STEPS - 1) as f32).round() as usize;
        palette[step.min(RAMP_STEPS - 1)].clone()
    }
}

/// A material of one colour, with the finish every actor here uses.
///
/// Scientific data has no artistic look to fall back on, so the surface is a
/// plain dielectric: shape has to read from lighting alone rather than from
/// gloss. `base_color` takes linear, which is what both [`super::sample`] and
/// [`super::elements::colour`] return.
pub(crate) fn flat(colour: LinearRgba) -> StandardMaterial {
    StandardMaterial {
        base_color: colour.into(),
        perceptual_roughness: 0.55,
        ..default()
    }
}

/// One placed, coloured copy of a shared mesh.
///
/// The mesh is named per instance rather than once for the whole actor, because
/// a kind may draw copies of more than one: ball and stick is spheres at the
/// atoms *and* cylinders along the bonds, and both are instances of primitives
/// the whole app shares.
pub(crate) struct Instance {
    pub mesh: Handle<Mesh>,
    pub transform: Transform,
    pub material: Handle<StandardMaterial>,
}

/// What an instanced kind produces instead of a mesh and a material.
///
/// [`place_instances`] turns these into entities under each placement.
#[derive(Component)]
pub(crate) struct Instances(pub Vec<Instance>);

/// Marks the entities `place_instances` owns, so it can clear its own work
/// without disturbing anything else parented to a placement.
#[derive(Component)]
struct Instanced;

/// Spawns one entity per instance under every placement of the actor.
///
/// This is the instanced counterpart of copying a mesh handle. It is also the
/// expensive system in this backend: a quarter-million-point cloud is a
/// quarter-million entities per placement, respawned whenever the instances
/// change. Rebuilding only on `Changed<Instances>` is what keeps that off the
/// per-frame path — a camera move or a slider on another actor costs nothing
/// here.
fn place_instances(
    mut commands: Commands,
    actors: Query<(Entity, &Instances, &Placements), Changed<Instances>>,
    existing: Query<(Entity, &ChildOf), With<Instanced>>,
) {
    for (actor, instances, placements) in &actors {
        for placement in placements.iter() {
            // Clear what this system put here last time. The placement's own
            // components are untouched.
            for (entity, parent) in &existing {
                if parent.parent() == placement {
                    commands.entity(entity).despawn();
                }
            }
            for instance in &instances.0 {
                commands.spawn((
                    Instanced,
                    instance.transform,
                    Mesh3d(instance.mesh.clone()),
                    RaytracingMesh3d(instance.mesh.clone()),
                    MeshMaterial3d(instance.material.clone()),
                    ChildOf(placement),
                ));
            }
        }
        debug!(
            "draw: solari placed {} instances of actor {actor} under {} object(s)",
            instances.0.len(),
            placements.len()
        );
    }
}

/// Gives every placement the mesh and material of a non-instanced actor.
///
/// The same job as the default backend's copier, plus `RaytracingMesh3d`: a
/// raytraced entity needs the handle twice, once for rasterised passes and once
/// for the acceleration structure.
fn copy_surfaces(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), Without<Instances>>,
    placements: Query<(Entity, &Placement, Option<&Mesh3d>)>,
) {
    for (entity, placement, current) in &placements {
        let Ok((mesh, material)) = actors.get(placement.0) else {
            continue;
        };
        if current.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert((
                mesh.clone(),
                RaytracingMesh3d(mesh.0.clone()),
                material.clone(),
            ));
        }
    }
}
