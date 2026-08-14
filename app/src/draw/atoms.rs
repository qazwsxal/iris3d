//! Balls and sticks, merged into one vertex buffer.
//!
//! The tessellation only — which triangles a set of atoms and bonds becomes.
//! Both the standard pipeline and the moment pathway want exactly the same ones
//! here: an opaque ball-and-stick is an opaque ball-and-stick, and the only
//! thing that differs is what each backend hangs off the resulting mesh.
//!
//! That is a deliberate exception to the note in [`super::elements`]
//! that tessellation belongs to a backend, and the same exception the cartoon
//! makes. The note is right when the backends genuinely differ — `rt`
//! *instances* one sphere rather than merging, and shares nothing with this —
//! and here two of the three do not.
//!
//! Cost is that the mesh scales with atom count: an icosphere is about 42
//! vertices, so a protein still wants impostor spheres rather than real
//! geometry. See [`Layout`] for what makes recolouring cheap despite that.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use super::elements;

/// Where each atom and bond ended up in the merged vertex buffer.
///
/// Written when the geometry is built, so a later colour change can find the
/// vertices belonging to atom *n* without rebuilding anything. This is the only
/// bookkeeping an in-place repaint needs, and it is what stops a colour-map drag
/// re-tessellating every sphere and cylinder.
#[derive(Component, Debug)]
pub struct Layout {
    /// Vertices in one atom's sphere.
    atom_vertices: usize,
    /// Vertices in one bond's cylinder.
    bond_vertices: usize,
    atoms: usize,
    /// Endpoint atom indices of the bonds actually drawn, in the order drawn.
    /// Bonds referring to missing atoms are skipped, so this is not simply the
    /// input bond list.
    bonds: Vec<[u32; 2]>,
}

impl Layout {
    pub fn vertices(&self) -> usize {
        self.atoms * self.atom_vertices + self.bonds.len() * self.bond_vertices
    }

    /// Expands per-atom colours to per-vertex, giving each bond the mean of its
    /// endpoints so the mapping stays continuous along a chain.
    pub fn colours(
        &self,
        atom_colours: &[[f32; 4]],
        stick: [f32; 4],
        tinted: bool,
    ) -> Vec<[f32; 4]> {
        let mut colours = Vec::with_capacity(self.vertices());
        for atom in 0..self.atoms {
            let colour = atom_colours.get(atom).copied().unwrap_or(stick);
            colours.extend(std::iter::repeat_n(colour, self.atom_vertices));
        }
        for [a, b] in &self.bonds {
            let colour = if tinted {
                mean(
                    atom_colours.get(*a as usize).copied().unwrap_or(stick),
                    atom_colours.get(*b as usize).copied().unwrap_or(stick),
                )
            } else {
                stick
            };
            colours.extend(std::iter::repeat_n(colour, self.bond_vertices));
        }
        colours
    }
}

pub fn mean(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let mut mean = [0.0; 4];
    for channel in 0..4 {
        mean[channel] = (a[channel] + b[channel]) * 0.5;
    }
    mean
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
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
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
pub struct Merged {
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

    pub fn vertices(&self) -> usize {
        self.positions.len()
    }

    pub fn triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Turns the accumulated geometry into a mesh.
    ///
    /// `with_colours` is what lets the moment pathway skip the colour buffer
    /// when it is drawing an absorbing medium that has no use for one.
    pub fn build(self, with_colours: bool) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        if with_colours {
            mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colours);
        }
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

/// What the tessellation needs to know beyond the atoms themselves.
pub struct Sizes {
    /// Multiplies each element's covalent radius.
    pub atom_scale: f32,
    /// Cylinder radius in ångströms, independent of the atoms.
    pub bond_radius: f32,
}

/// Merges one sphere per atom and one cylinder per bond into a single buffer.
///
/// `bonds` are index pairs into `positions`, already narrowed and renumbered by
/// whatever subset applies — a bond naming an atom that is not here is dropped,
/// because half a bond sticking out into space reads as broken geometry rather
/// than as a deliberate cut.
///
/// Returns `None` only when the primitives themselves cannot be read, which
/// would be a Bevy problem rather than a data one.
pub fn build(
    positions: &[Vec3],
    elements: &[u32],
    bonds: &[u32],
    atom_colours: &[[f32; 4]],
    stick: [f32; 4],
    tinted: bool,
    sizes: &Sizes,
) -> Option<(Merged, Layout)> {
    let (sphere, cylinder) = (
        Template::from_mesh(&Sphere::new(1.0).mesh().ico(2).ok()?)?,
        Template::from_mesh(&Cylinder::new(1.0, 1.0).mesh().resolution(10).build())?,
    );

    let mut merged = Merged::default();
    for (index, position) in positions.iter().enumerate() {
        let atomic_number = elements.get(index).copied().unwrap_or(6);
        let radius = elements::radius(atomic_number) * sizes.atom_scale.max(0.01);
        merged.stamp(
            &sphere,
            &Transform::from_translation(*position).with_scale(Vec3::splat(radius)),
            atom_colours.get(index).copied().unwrap_or(stick),
        );
    }

    let mut drawn: Vec<[u32; 2]> = Vec::new();
    for pair in bonds.chunks_exact(2) {
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
        // When atoms are field-coloured, a bond takes the mean of its endpoints
        // so the mapping stays continuous along the chain.
        let colour = if tinted {
            mean(
                atom_colours.get(pair[0] as usize).copied().unwrap_or(stick),
                atom_colours.get(pair[1] as usize).copied().unwrap_or(stick),
            )
        } else {
            stick
        };
        merged.stamp(
            &cylinder,
            // Bevy's cylinder runs along +Y, so rotate that onto the bond.
            &Transform::from_translation((*a + *b) * 0.5)
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, along / length))
                .with_scale(Vec3::new(sizes.bond_radius, length, sizes.bond_radius)),
            colour,
        );
        drawn.push([pair[0], pair[1]]);
    }

    let layout = Layout {
        atom_vertices: sphere.positions.len(),
        bond_vertices: cylinder.positions.len(),
        atoms: positions.len(),
        bonds: drawn,
    };
    Some((merged, layout))
}
