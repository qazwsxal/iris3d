//! Triangles drawn as an opaque, lit surface.
//!
//! It goes through Bevy's ordinary passes, writes depth, and is what the
//! absorbance of anything in front of it is measured against. Geometry in, a lit
//! shape out — the least surprising thing an id can mean, and the same thing
//! under every pathway.
//!
//! # It builds nothing
//!
//! One `geometry` input, and what it does with it is `Mesh3d(handle.clone())`.
//! The vertices were assembled once by whatever produced the geometry — see
//! [`filter::geometry`](crate::filter::geometry) — and this kind adds a material
//! and a pass. That is the whole difference between it and
//! [`medium`](super::medium): same handle, same buffers, two materials.
//!
//! Assembling the mesh here instead would mean two actors over one ribbon
//! uploading the same vertices twice, which is what the split into filters sets
//! out to stop.
//!
//! # The name
//!
//! `surface` is what ParaView, VTK, MayaVi and PyMOL all call this — an opaque
//! filled representation of triangles — so it is the one word a person arriving
//! from any of them already knows.
//!
//! It was called `mesh` for a while, to keep `surface` free for a *molecular*
//! surface. That reservation is void: in PyMOL and ChimeraX `mesh` means the
//! **wireframe**, so the name was borrowed from the representation iris3d still
//! intends to add, and `mesh` collided with the `Mesh` being passed around
//! besides. A solvent-excluded surface is a **filter** now that generating
//! geometry and displaying it are separate jobs — `ses`, `sas` — and what draws
//! its triangles is this. PyMOL's `show surface` conflates the two because it
//! has to; iris3d does not.
//!
//! For a closed mesh drawn as the *medium it bounds* — thickness you can see
//! through, optionally with a glass skin — see [`medium`](super::medium).

use bevy::prelude::*;

use crate::scene::DataStore;
use crate::scene::registry::{ActorKind, ActorRegistry};
use iris3d_data::array::StoredGeometry;
use iris3d_model::{Bindings, ParamKind, ParamSpec, flag};

use crate::scene::link::Placement;

use super::{Actor, Dirty, mark};

/// What this pathway needs to redraw a mesh: the geometry it was given and an
/// ordinary material, since an opaque surface takes no part in the moment
/// passes.
type Drawable<'a> = (
    Actor<'a, SurfaceStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<StandardMaterial>>,
);

/// Triangles, shaded and opaque.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct SurfaceStyle {
    /// Light and draw back faces as well as front ones.
    ///
    /// On by default because scientific meshes are routinely open or have
    /// inconsistent winding, and a one-sided material renders those as holes.
    /// Turning it off is how you see through to the inside of a closed mesh.
    pub double_sided: bool,
    /// Linear RGB, used where the geometry carries no colours of its own.
    pub tint: Vec3,
}

const PARAMS: &[ParamSpec] = &[
    // One input. Positions, triangles, normals and per-vertex colour arrive
    // together as a mesh somebody else assembled — the `geometry` filter, for an
    // upload as much as for a computed ribbon, so there is no second path.
    ParamSpec {
        id: "geometry",
        label: "geometry",
        kind: ParamKind::Geometry { required: true },
    },
    crate::draw::TINT,
    ParamSpec {
        id: "double_sided",
        label: "double sided",
        kind: ParamKind::Bool { default: true },
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "surface",
        label: "surface",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(SurfaceStyle {
                double_sided: flag(params, "double_sided", true),
                tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
            });
        },
    });
}

/// `double_sided` is a material property; nothing about the mesh depends on it.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<SurfaceStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
}

pub fn draw_surfaces(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, bound, dirty), mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Some(geometry) = geometry(bound, &store) else {
            // Required, so `check_bindings` refused an actor without it;
            // reaching here means the geometry was released underneath one.
            continue;
        };

        // Shared, not copied. Two actors bound to one geometry hold the same
        // asset id, and asserting that is what proves nothing was duplicated.
        if mesh3d.map(|Mesh3d(handle)| handle.id()) != Some(geometry.handle.id()) {
            commands
                .entity(entity)
                .insert(Mesh3d(geometry.handle.clone()));
        }

        // Whether the geometry carries colours decides the base colour, so a run
        // that adds or drops them has to rewrite the material even though
        // nothing about this actor's own settings moved. That is why a changed
        // geometry asset marks the material dirty; see `draw::mark_dirty`.
        ensure_material(
            &mut commands,
            entity,
            &mut materials,
            material3d,
            StandardMaterial {
                // Vertex colours multiply the base, so it has to be white for
                // them to come through unaltered.
                base_color: match geometry.meta.colours {
                    true => Color::WHITE,
                    false => Color::linear_rgb(style.tint.x, style.tint.y, style.tint.z),
                },
                perceptual_roughness: 0.55,
                double_sided: style.double_sided,
                // `double_sided` only lights the back faces; they still have to
                // survive culling to be lit at all.
                cull_mode: if style.double_sided {
                    None
                } else {
                    Some(bevy::render::render_resource::Face::Back)
                },
                ..default()
            },
        );
    }
}

/// The geometry an actor binds, or `None` if nothing resolves.
///
/// The counterpart of [`bound`](crate::draw::bound) for a mesh rather than an
/// array. `None` covers a released handle and one that names an array — which
/// `check_bindings` refuses, so it only happens if that binding was released and
/// the id reused.
pub(super) fn geometry<'a>(
    bindings: &Bindings,
    store: &'a DataStore,
) -> Option<&'a StoredGeometry> {
    store.geometry(bindings.get("geometry")?)
}

/// Replaces a material in place when the actor already has one, so a rebuild
/// does not leak a fresh `StandardMaterial` into `Assets`.
fn ensure_material(
    commands: &mut Commands,
    entity: Entity,
    materials: &mut Assets<StandardMaterial>,
    existing: Option<&MeshMaterial3d<StandardMaterial>>,
    material: StandardMaterial,
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

/// Gives every placement the mesh and material the actor holds.
#[allow(clippy::type_complexity)]
pub fn place_surfaces(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<SurfaceStyle>>,
    placements: Query<(
        Entity,
        &Placement,
        Option<&Mesh3d>,
        Option<&MeshMaterial3d<StandardMaterial>>,
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
