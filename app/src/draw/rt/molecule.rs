//! Ball and stick, instanced.
//!
//! A sphere per atom and a cylinder per bond, all instancing the two meshes in
//! [`Primitives`]. The default backend merges the same shapes into one mesh; a
//! raytraced scene wants the opposite, because one mesh means one acceleration
//! structure however many copies reference it.
//!
//! Colour is per atom, so it comes from the material palette rather than from
//! vertices. Element colouring is the usual case here and needs no palette at
//! all — CPK colours are a fixed table, not a ramp, so each distinct element
//! gets one material.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::draw::elements;
use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, float};
use crate::scene::{DataArray, DataStore, Dtype};

use super::{Actor, Dirty, Instance, Instances, Primitives, RampPalette, bound, flat, mark, normalised};

/// Atom radii and bond thickness are geometry: both change where instances go.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct BallAndStickStyle {
    /// Multiplies each element's covalent radius.
    pub atom_scale: f32,
    /// Cylinder radius in ångströms, independent of the atoms.
    pub bond_radius: f32,
}

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "atom centres",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
        },
    },
    ParamSpec {
        id: "elements",
        label: "elements",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: false,
        },
    },
    ParamSpec {
        id: "bonds",
        label: "bonds",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 2],
            required: false,
        },
    },
    ParamSpec {
        id: "colour",
        label: "colour by",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
        },
    },
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
        // A convention about what a molecule looks like, not about how the
        // spheres and cylinders reach the screen.
        shared: true,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(BallAndStickStyle {
                atom_scale: float(params, "atom_scale", 0.25),
                bond_radius: float(params, "bond_radius", 0.1),
            });
        },
    });
}

pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<BallAndStickStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

/// The transform putting a unit cylinder along the segment `from`..`to`.
///
/// The primitive runs along +Y with unit height, centred on the origin, so this
/// is a rotation from +Y onto the bond direction, a scale to its length and
/// radius, and a translation to its midpoint.
fn along(from: Vec3, to: Vec3, radius: f32) -> Option<Transform> {
    let axis = to - from;
    let length = axis.length();
    if length < f32::EPSILON {
        return None;
    }
    Some(
        Transform::from_translation(from.midpoint(to))
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, axis / length))
            .with_scale(Vec3::new(radius, length, radius)),
    )
}

pub fn draw_molecules(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut palette: ResMut<RampPalette>,
    primitives: Res<Primitives>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Actor<BallAndStickStyle>>,
) {
    for (entity, style, colour, _subset, bindings, dirty) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Some(position_array) = bound(bindings, "positions", &store, &arrays) else {
            continue;
        };
        let centres = position_array.to_vec3();
        if centres.is_empty() {
            continue;
        }

        let atomic_numbers: Vec<u32> = bound(bindings, "elements", &store, &arrays)
            .and_then(|array| array.to_u32())
            .unwrap_or_default();
        let element_of = |atom: usize| atomic_numbers.get(atom).copied().unwrap_or(6);

        // A bound colour array wins; otherwise atoms take their element colour,
        // which is what a molecule looks like by default.
        let ramp = bound(bindings, "colour", &store, &arrays)
            .and_then(|values| normalised(values, colour, position_array.count() as usize));

        // One material per distinct element, built on demand. A protein has a
        // handful of elements and tens of thousands of atoms, so this is a very
        // short table.
        let mut by_element: HashMap<u32, Handle<StandardMaterial>> = HashMap::new();
        let mut atom_material = |atom: usize,
                                 palette: &mut RampPalette,
                                 materials: &mut Assets<StandardMaterial>| {
            match &ramp {
                Some(ramp) => palette.pick(colour.map, materials, ramp[atom]),
                None => by_element
                    .entry(element_of(atom))
                    .or_insert_with(|| {
                        let rgba = elements::colour(element_of(atom));
                        materials.add(flat(LinearRgba::from_f32_array(rgba)))
                    })
                    .clone(),
            }
        };

        let mut items = Vec::with_capacity(centres.len());
        for (atom, centre) in centres.iter().enumerate() {
            let radius = elements::radius(element_of(atom)) * style.atom_scale;
            items.push(Instance {
                mesh: primitives.sphere.clone(),
                transform: Transform::from_translation(*centre).with_scale(Vec3::splat(radius)),
                material: atom_material(atom, &mut palette, &mut materials),
            });
        }

        // Bonds take the colour of the atom they start from. Splitting each in
        // half to take both would double the cylinder count for a subtlety that
        // does not read at bond thickness.
        if let Some(bonds) = bound(bindings, "bonds", &store, &arrays).and_then(|a| a.to_u32()) {
            for pair in bonds.chunks_exact(2) {
                let (a, b) = (pair[0] as usize, pair[1] as usize);
                if a >= centres.len() || b >= centres.len() {
                    warn!("draw: bond {a}-{b} exceeds {} atoms", centres.len());
                    continue;
                }
                let Some(transform) = along(centres[a], centres[b], style.bond_radius) else {
                    continue;
                };
                items.push(Instance {
                    mesh: primitives.cylinder.clone(),
                    transform,
                    material: atom_material(a, &mut palette, &mut materials),
                });
            }
        }

        debug!(
            "draw: solari ball and stick, {} atoms and {} instances",
            centres.len(),
            items.len()
        );
        commands.entity(entity).insert(Instances(items));
    }
}
