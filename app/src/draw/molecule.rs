//! Ball-and-stick molecules, merged into a single mesh.
//!
//! One sphere per atom and one cylinder per bond, but baked into one vertex
//! buffer with per-vertex colours rather than an entity each. Atom count no
//! longer drives entity count, and — because the whole representation is a
//! single `Mesh3d` on the representation entity — rebuilding is just replacing
//! that mesh, with no child entities to reconcile or leak.
//!
//! Cost is that the mesh scales with atom count: an icosphere per atom is
//! roughly 42 vertices, so a protein still wants impostor spheres rather than
//! real geometry.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use crate::scene::{ColorBy, DataArray, MoleculeData, Representation};

use super::NeedsRedraw;

/// Covalent radii by atomic number, in ångströms, for the common elements.
/// Anything unlisted falls back to carbon.
fn element_radius(atomic_number: u32) -> f32 {
    match atomic_number {
        1 => 0.31,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        _ => 0.76,
    }
}

/// Standard CPK colouring.
///
/// Returned linear, not sRGB: vertex colours go to the shader untouched and
/// `pbr_fragment.wgsl` assigns them straight to `base_color`, so writing sRGB
/// values here renders everything far too bright.
fn element_colour(atomic_number: u32) -> [f32; 4] {
    let rgb = match atomic_number {
        1 => [0.95, 0.95, 0.95],
        6 => [0.25, 0.25, 0.28],
        7 => [0.19, 0.31, 0.97],
        8 => [0.94, 0.15, 0.10],
        9 => [0.56, 0.88, 0.31],
        15 => [1.00, 0.50, 0.00],
        16 => [0.90, 0.78, 0.19],
        17 => [0.12, 0.94, 0.12],
        _ => [0.85, 0.45, 0.85],
    };
    Color::srgb(rgb[0], rgb[1], rgb[2])
        .to_linear()
        .to_f32_array()
}

/// Positions, normals and indices lifted out of a generated primitive so they
/// can be stamped into the merged buffer.
struct Template {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<u32>,
}

impl Template {
    fn from_mesh(mesh: &Mesh) -> Option<Self> {
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            return None;
        };
        let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            return None;
        };
        let indices = match mesh.indices()? {
            Indices::U32(values) => values.clone(),
            Indices::U16(values) => values.iter().map(|i| *i as u32).collect(),
        };
        Some(Self {
            positions: positions.iter().map(|p| Vec3::from(*p)).collect(),
            normals: normals.iter().map(|n| Vec3::from(*n)).collect(),
            indices,
        })
    }
}

/// Accumulates transformed copies of templates into one mesh.
#[derive(Default)]
struct Merged {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colours: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Merged {
    fn stamp(&mut self, template: &Template, transform: &Transform, colour: [f32; 4]) {
        let base = self.positions.len() as u32;
        let matrix = transform.to_matrix();
        // Normals need the inverse transpose; scaling here is per-axis for
        // cylinders, so simply rotating the normal would be wrong.
        let normal_matrix = matrix.inverse().transpose();

        for (position, normal) in template.positions.iter().zip(&template.normals) {
            let world = matrix.transform_point3(*position);
            let n = (normal_matrix.transform_vector3(*normal)).normalize_or_zero();
            self.positions.push([world.x, world.y, world.z]);
            self.normals.push([n.x, n.y, n.z]);
            self.colours.push(colour);
        }
        self.indices
            .extend(template.indices.iter().map(|index| base + index));
    }

    fn build(self) -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colours);
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

pub fn draw_molecules(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    dirty: Query<(Entity, &Representation, &ColorBy, &ChildOf), With<NeedsRedraw>>,
    molecules: Query<&MoleculeData>,
) {
    for (entity, representation, colour, parent) in &dirty {
        let Representation::BallAndStick {
            atom_scale,
            bond_radius,
        } = representation
        else {
            continue;
        };
        let Ok(molecule) = molecules.get(parent.parent()) else {
            continue;
        };
        let Some(position_array) = arrays.get(&molecule.positions) else {
            continue;
        };

        let positions = position_array.to_vec3();
        if positions.is_empty() {
            continue;
        }

        let elements: Vec<u32> = molecule
            .elements
            .as_ref()
            .and_then(|handle| arrays.get(handle))
            .and_then(|array| array.to_u32())
            .unwrap_or_else(|| vec![6; positions.len()]);

        let (Some(sphere), Some(cylinder)) = (
            Template::from_mesh(&Sphere::new(1.0).mesh().ico(2).unwrap()),
            Template::from_mesh(&Cylinder::new(1.0, 1.0).mesh().resolution(10).build()),
        ) else {
            warn!("draw: could not read primitive templates for a molecule");
            continue;
        };

        let mut merged = Merged::default();

        for (index, position) in positions.iter().enumerate() {
            let atomic_number = elements.get(index).copied().unwrap_or(6);
            let radius = element_radius(atomic_number) * atom_scale.max(0.01);
            merged.stamp(
                &sphere,
                &Transform::from_translation(*position).with_scale(Vec3::splat(radius)),
                element_colour(atomic_number),
            );
        }

        let stick = colour.flat.to_linear().to_f32_array();
        let mut sticks = 0usize;
        if let Some(bonds) = &molecule.bonds {
            let pairs = arrays
                .get(&bonds.pairs)
                .and_then(|array| array.to_u32())
                .unwrap_or_default();
            for pair in pairs.chunks_exact(2) {
                let (Some(a), Some(b)) = (
                    positions.get(pair[0] as usize),
                    positions.get(pair[1] as usize),
                ) else {
                    continue;
                };
                let along = *b - *a;
                let length = along.length();
                if length < f32::EPSILON {
                    continue;
                }
                merged.stamp(
                    &cylinder,
                    // Bevy's cylinder runs along +Y, so rotate that onto the bond.
                    &Transform::from_translation((*a + *b) * 0.5)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, along / length))
                        .with_scale(Vec3::new(*bond_radius, length, *bond_radius)),
                    stick,
                );
                sticks += 1;
            }
        }

        let vertices = merged.positions.len();
        commands.entity(entity).insert((
            Mesh3d(meshes.add(merged.build())),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.4,
                ..default()
            })),
        ));

        debug!(
            "draw: molecule merged into one mesh — {} atoms, {sticks} bonds, {vertices} vertices",
            positions.len()
        );
    }
}
