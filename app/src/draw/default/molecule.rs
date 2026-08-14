//! Ball-and-stick molecules, opaque, composing with the volumes around them.
//!
//! The tessellation is [`crate::draw::atoms`]. What is here is where it sits:
//! the atoms are opaque, so they write depth, and the moment accumulation
//! truncates every interval at that depth.
//!
//! That is what this kind is *for*. At the resolution where a map shows
//! individual atoms — 1.7 A and better — the question stops being "does the
//! model fit the envelope" and becomes "is this side chain in its density", and
//! answering it needs the density in front of an atom to dim it while the
//! density behind does not. A cartoon cannot answer it at all, because it draws
//! no side chains.
//!
//! # Opaque only
//!
//! As for [`glycan`](super::glycan), and unlike [`cartoon`](super::cartoon):
//! there is no absorbing mode. A ball-and-stick is a *ball* — a hard little
//! sphere standing for an atom — and making it a medium light passes through
//! says something about atoms that nobody means.

use bevy::prelude::*;

use crate::draw::atoms::{self, Layout, Sizes};
use crate::scene::link::Placement;
use crate::scene::registry::{
    ActorKind, ActorRegistry, Bindings, ParamKind, ParamSpec, float,
};
use crate::scene::subset::Remap;
use crate::scene::{DataArray, DataStore, Dtype, Subset};

use super::{Actor, Dirty, bound, mark};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "atom centres",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "elements",
        label: "elements",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    // No bonds means balls and no sticks, which is the honest way to draw a
    // structure whose connectivity nobody computed.
    ParamSpec {
        id: "bonds",
        label: "bonds",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 2],
            required: false,
            structural: true,
        },
    },
    // Linear RGB, one triple per atom, already mapped. Unbound falls back to CPK
    // colours from the periodic table rather than to `tint`, because that is
    // what a molecule with no field on it should look like.
    //
    // `structural`, unlike the same input on `mesh` and `points`. Not because a
    // colour is structural here, but because this kind has no repaint path from
    // a *rebuild* decision: `read` runs first and `dirty.geometry` decides the
    // rest. Making a bound-array change repaint is worth doing — a merged
    // protein is exactly the case that hurts — and until then this says
    // truthfully what happens.
    ParamSpec {
        id: "colour",
        label: "colour",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: false,
            structural: true,
        },
    },
    crate::draw::TINT,
    ParamSpec {
        id: "atom_scale",
        label: "atom scale",
        kind: ParamKind::Float {
            default: 0.25,
            min: 0.05,
            max: 1.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "bond_radius",
        label: "bond radius",
        kind: ParamKind::Float {
            default: 0.1,
            min: 0.01,
            max: 0.5,
            logarithmic: false,
        },
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "ball-and-stick",
        label: "ball and stick",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(BallAndStickStyle {
                sizes: Sizes {
                    atom_scale: float(params, "atom_scale", 0.25),
                    bond_radius: float(params, "bond_radius", 0.1),
                },
                tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
            });
        },
    });
}

/// Radii, bond thickness and the stick colour, as this backend's style
/// component.
#[derive(Component)]
pub struct BallAndStickStyle {
    pub sizes: Sizes,
    /// Linear RGB for the bonds, and for every atom when nothing is bound to
    /// `colour`. Bonds join two atoms of possibly different elements, so they
    /// take one colour rather than a CPK one.
    pub tint: Vec3,
}

/// Both parameters change where vertices go, so both are geometry.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<BallAndStickStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

type Drawable<'a> = (
    Actor<'a, BallAndStickStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<StandardMaterial>>,
);

pub fn draw_molecules(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
    layouts: Query<&Layout>,
) {
    for ((entity, style, subset, bindings, dirty), mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Some(atoms) = read(bindings, subset, &store, &arrays, style.tint) else {
            continue;
        };

        // Nothing moved, so the buffer can be painted over rather than rebuilt.
        if !dirty.geometry {
            if let Ok(layout) = layouts.get(entity) {
                repaint(
                    &mut meshes,
                    mesh3d,
                    layout.colours(&atoms.colours, atoms.stick, atoms.tinted),
                );
            }
            continue;
        }

        let Some((merged, layout)) = atoms::build(
            &atoms.positions,
            &atoms.elements,
            &atoms.bonds,
            &atoms.colours,
            atoms.stick,
            atoms.tinted,
            &style.sizes,
        ) else {
            warn!("draw: could not read the primitives for a molecule");
            continue;
        };

        debug!(
            "draw: opaque ball-and-stick rebuilt — {} atoms, {} vertices, {} triangles",
            atoms.positions.len(),
            merged.vertices(),
            merged.triangles()
        );

        if let Some(Mesh3d(handle)) = mesh3d
            && let Some(mut slot) = meshes.get_mut(handle)
        {
            *slot = merged.build(true);
        } else {
            commands
                .entity(entity)
                .insert(Mesh3d(meshes.add(merged.build(true))));
        }
        commands.entity(entity).insert(layout);

        if material3d.is_none() {
            commands
                .entity(entity)
                .insert(MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 0.4,
                    ..default()
                })));
        }
    }
}

/// The decoded arrays plus the colours they imply.
struct Atoms {
    positions: Vec<Vec3>,
    elements: Vec<u32>,
    bonds: Vec<u32>,
    colours: Vec<[f32; 4]>,
    stick: [f32; 4],
    /// Whether the colours came from a bound array rather than from CPK.
    tinted: bool,
}

/// Reads what the actor bound, narrowed to its subset.
///
/// Atoms are renumbered by a subset and bonds refer to atoms by index, so both
/// the positions and the bond list have to be rewritten rather than filtered.
fn read(
    bindings: &Bindings,
    subset: &Subset,
    store: &DataStore,
    arrays: &Assets<DataArray>,
    tint: Vec3,
) -> Option<Atoms> {
    let position_array = bound(bindings, "positions", store, arrays)?;
    let all = position_array.to_vec3();
    if all.is_empty() {
        return None;
    }
    // Carbon for everything when no elements are bound: radii and colours still
    // need a number each, and a structure with no element data is better drawn
    // uniformly than not drawn.
    let all_elements: Vec<u32> = bound(bindings, "elements", store, arrays)
        .and_then(|array| array.to_u32())
        .unwrap_or_else(|| vec![6; all.len()]);

    let kept = subset.selected(all.len(), arrays);
    let remap = kept.as_ref().map(|kept| Remap::new(kept, all.len()));
    let positions: Vec<Vec3> = match &kept {
        Some(kept) => kept.iter().map(|index| all[*index as usize]).collect(),
        None => all,
    };
    let elements: Vec<u32> = match &kept {
        Some(kept) => kept
            .iter()
            .filter_map(|index| all_elements.get(*index as usize).copied())
            .collect(),
        None => all_elements,
    };

    let bonds = bound(bindings, "bonds", store, arrays)
        .and_then(|array| array.to_u32())
        .map(|pairs| match &remap {
            Some(remap) => pairs
                .chunks_exact(2)
                .filter_map(|pair| {
                    let (a, b) = (remap.get(pair[0])?, remap.get(pair[1])?);
                    Some([a, b])
                })
                .flatten()
                .collect(),
            None => pairs,
        })
        .unwrap_or_default();

    // A bound colour array wins over CPK: an input that is bound has to
    // actually apply, or the tree claims a colouring the render does not show.
    let bound_rgb = bound(bindings, "colour", store, arrays)
        .and_then(|values| super::super::bound_colours(values, position_array.count() as usize))
        .map(|colours| match &kept {
            Some(kept) => kept.iter().map(|index| colours[*index as usize]).collect(),
            None => colours,
        });
    let stick = tint.extend(1.0).to_array();
    let tinted = bound_rgb.is_some();
    let colours: Vec<[f32; 4]> = (0..positions.len())
        .map(|index| {
            bound_rgb.as_ref().map_or_else(
                || crate::draw::elements::colour(elements.get(index).copied().unwrap_or(6)),
                |colours| colours[index],
            )
        })
        .collect();

    Some(Atoms {
        positions,
        elements,
        bonds,
        colours,
        stick,
        tinted,
    })
}

fn repaint(meshes: &mut Assets<Mesh>, existing: Option<&Mesh3d>, colours: Vec<[f32; 4]>) {
    let Some(Mesh3d(handle)) = existing else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(handle) else {
        return;
    };
    if mesh.count_vertices() != colours.len() {
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

/// Gives every placement the mesh and material the actor holds. As for
/// [`glycan::place_glycans`](super::glycan::place_glycans): these carry no
/// `MomentVolume`, so the pathway's own placement system does not match them.
#[allow(clippy::type_complexity)]
pub fn place_molecules(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<BallAndStickStyle>>,
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
