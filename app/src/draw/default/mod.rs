//! The default backend: Bevy's standard rendering pipeline.
//!
//! One `Mesh3d` and one `Material` per actor, handed to the renderer Bevy ships
//! with. It is deliberately the simple option — it gets every sample dataset on
//! screen and gives a reference image to check more ambitious pathways against.
//! It will not scale; see the notes on each actor kind for where it gives out.
//!
//! What makes this a backend rather than a library of drawing code is the
//! choice of output. Everything here assumes an actor's GPU data is a mesh plus
//! a material, and that the renderer will sort and composite it. A pathway that
//! accumulates moments, or one that builds acceleration structures, wants a
//! different mapping from the same CPU arrays — which is why the actor kinds
//! belong to the backend rather than sitting above it. Nothing a client sends
//! mentions any of this: it binds arrays and sets parameters, and what those
//! become on the GPU is not part of the contract.
//!
//! The shared half — invalidation, binding resolution and colour maps — is
//! [`super`], and is re-exported here so a kind module can reach either through
//! the same path.

use bevy::asset::embedded_asset;
use bevy::prelude::*;

use crate::scene::link::Placement;
use crate::scene::registry::ActorRegistry;

// Re-exported rather than imported separately in each kind module: whether a
// helper is shared or belongs to this backend is a fact about the pathway, not
// something four call sites should have to track.
pub(crate) use super::{Actor, Dirty, Draw, Invalidate, Place, bound, bound_colours, mark};

mod molecule;
mod points;
mod surface;
mod volume;

use points::PointQuadMaterial;
use volume::VolumeMaterial;

/// What this backend needs to redraw one actor: everything any backend needs,
/// plus whatever it produced last time.
///
/// The tail is the part that makes this the default backend's query and not
/// some other's — a mesh and a material are what *this* pipeline draws.
/// Carrying the previous handles is what makes reuse rather than reallocation
/// possible: a rebuild writes into the asset the actor already holds, so
/// dragging a slider allocates nothing.
pub(crate) type Drawable<'a, Style, Material> = (
    Actor<'a, Style>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<Material>>,
);

/// Replaces a mesh in place when the actor already has one.
///
/// Reusing the handle keeps the entity pointing at the same asset — and, more
/// to the point, stops every rebuild leaking a fresh `Mesh` into `Assets`.
pub(crate) fn ensure_mesh(
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
pub(crate) fn ensure_material<M: Material>(
    commands: &mut Commands,
    entity: Entity,
    materials: &mut Assets<M>,
    existing: Option<&MeshMaterial3d<M>>,
    material: M,
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

/// Overwrites a mesh's vertex colours without touching anything else.
///
/// Only legal when the vertex count is unchanged, which is exactly when the
/// geometry is not also dirty.
pub(crate) fn repaint(
    meshes: &mut Assets<Mesh>,
    existing: Option<&Mesh3d>,
    colours: Vec<[f32; 4]>,
) {
    let Some(Mesh3d(handle)) = existing else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(handle) else {
        return;
    };
    if mesh.count_vertices() != colours.len() {
        // Should not happen, and silently painting a prefix would be worse than
        // waiting for the rebuild that is evidently coming.
        warn!(
            "draw: {} vertex colours for a mesh of {} vertices; skipping the repaint",
            colours.len(),
            mesh.count_vertices()
        );
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

pub struct DefaultBackendPlugin;

impl Plugin for DefaultBackendPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "point_quad.wgsl");
        embedded_asset!(app, "volume.wgsl");

        // Registration order is presentation order only — what the UI lists and
        // what `ListActorKinds` returns. Nothing picks a kind on a caller's
        // behalf.
        {
            let mut registry = app.world_mut().resource_mut::<ActorRegistry>();
            points::register(&mut registry);
            surface::register(&mut registry);
            molecule::register(&mut registry);
            volume::register(&mut registry);
        }

        app.add_plugins((
            MaterialPlugin::<PointQuadMaterial>::default(),
            MaterialPlugin::<VolumeMaterial>::default(),
        ))
        .add_systems(
            Update,
            (
                // Each kind classifies its own style changes: only it knows
                // whether one of its parameters feeds the geometry or a
                // material uniform, and centralising that would put the kind's
                // knowledge back in shared code.
                (
                    points::invalidate,
                    surface::invalidate,
                    molecule::invalidate,
                    volume::invalidate,
                )
                    .in_set(Invalidate),
                (
                    points::draw_points,
                    surface::draw_surfaces,
                    molecule::draw_molecules,
                    volume::draw_volumes,
                )
                    .in_set(Draw),
                // No `VolumeMaterial` here: a volume's uniform holds the world
                // transform of the copy being drawn, so its placements each get
                // their own material rather than a copy of one. See `volume`.
                (
                    copy_meshes,
                    copy_materials::<StandardMaterial>,
                    copy_materials::<PointQuadMaterial>,
                )
                    .in_set(Place),
            ),
        );
    }
}

/// Gives every placement of an actor the mesh that actor owns.
///
/// The handle, not the geometry. A kind rebuilds into the asset it already
/// holds, so the copies never need touching again — this only has to run when
/// an actor's handle is first created or genuinely replaced, and the comparison
/// makes that the common case. Several placements sharing one mesh and one
/// material is also what lets Bevy batch them into a single draw.
fn copy_meshes(
    mut commands: Commands,
    actors: Query<&Mesh3d>,
    placements: Query<(Entity, &Placement, Option<&Mesh3d>)>,
) {
    for (entity, placement, current) in &placements {
        let Ok(mesh) = actors.get(placement.0) else {
            // The actor has not been drawn yet. Nothing to copy, and it will be
            // here next frame.
            continue;
        };
        if current.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert(mesh.clone());
        }
    }
}

/// As [`copy_meshes`], for the material.
///
/// Generic because the material type is the kind's choice, so this is
/// registered once per material this backend knows about.
fn copy_materials<M: Material>(
    mut commands: Commands,
    actors: Query<&MeshMaterial3d<M>>,
    placements: Query<(Entity, &Placement, Option<&MeshMaterial3d<M>>)>,
) {
    for (entity, placement, current) in &placements {
        let Ok(material) = actors.get(placement.0) else {
            continue;
        };
        if current.map(|MeshMaterial3d(handle)| handle.id()) != Some(material.0.id()) {
            commands.entity(entity).insert(material.clone());
        }
    }
}
