//! Ball-and-stick molecules, merged into a single mesh.
//!
//! One sphere per atom and one cylinder per bond, but baked into one vertex
//! buffer with per-vertex colours rather than an entity each. Atom count no
//! longer drives entity count, and — because the whole actor is a single
//! `Mesh3d` on the actor entity — rebuilding is just replacing that mesh, with
//! no child entities to reconcile or leak.
//!
//! Cost is that the mesh scales with atom count: an icosphere per atom is
//! roughly 42 vertices, so a protein still wants impostor spheres rather than
//! real geometry.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, float};
use crate::scene::subset::Remap;
use crate::scene::{DataArray, DataStore, Dtype};

use super::{Dirty, Drawable, mark};

/// Where each atom and bond ended up in the merged vertex buffer.
///
/// Written when the geometry is built, so a later colour change can find the
/// vertices belonging to atom *n* without rebuilding anything. This is the only
/// bookkeeping the whole in-place repaint needs, and it is what stops a
/// colour-map drag re-tessellating every sphere and cylinder in a protein.
#[derive(Component, Debug)]
pub struct MoleculeLayout {
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

impl MoleculeLayout {
    fn vertices(&self) -> usize {
        self.atoms * self.atom_vertices + self.bonds.len() * self.bond_vertices
    }

    /// Expands per-atom colours to per-vertex, giving each bond the mean of its
    /// endpoints so the mapping stays continuous along a chain.
    fn colours(&self, atom_colours: &[[f32; 4]], stick: [f32; 4], tinted: bool) -> Vec<[f32; 4]> {
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

fn mean(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let mut mean = [0.0; 4];
    for channel in 0..4 {
        mean[channel] = (a[channel] + b[channel]) * 0.5;
    }
    mean
}

/// Atom radii and bond thickness are geometry: both change where vertices go.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<BallAndStickStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

/// Spheres at atoms, cylinders along bonds.
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
    // Atomic numbers, which drive radii and CPK colouring. An "elements" buffer
    // is what made an upload a molecule under inference; here it is simply what
    // this kind needs in order to know an atom from another.
    ParamSpec {
        id: "elements",
        label: "elements",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: false,
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
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(BallAndStickStyle {
                atom_scale: float(params, "atom_scale", 0.25),
                bond_radius: float(params, "bond_radius", 0.1),
            });
        },
    });
}

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
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
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
    store: Res<DataStore>,
    dirty: Query<Drawable<BallAndStickStyle, StandardMaterial>>,
    layouts: Query<&MoleculeLayout>,
) {
    for (entity, style, colour, subset, bound, dirty, mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let layout = layouts.get(entity).ok();
        let BallAndStickStyle {
            atom_scale,
            bond_radius,
        } = style;
        let Some(position_array) = super::bound(bound, "positions", &store, &arrays) else {
            continue;
        };

        let all = position_array.to_vec3();
        if all.is_empty() {
            continue;
        }

        // Carbon for everything when no elements are bound: radii and colours
        // still need a number each, and a structure with no element data is
        // better drawn uniformly than not drawn.
        let all_elements: Vec<u32> = super::bound(bound, "elements", &store, &arrays)
            .and_then(|array| array.to_u32())
            .unwrap_or_else(|| vec![6; all.len()]);

        // Atoms are renumbered by a subset, and bonds refer to atoms by index,
        // so both the positions and the bond list have to be rewritten.
        let kept = subset.selected(all.len(), &arrays);
        let remap = kept.as_ref().map(|kept| Remap::new(kept, all.len()));
        let narrow = |values: &[u32]| -> Vec<u32> {
            match &kept {
                Some(kept) => kept.iter().map(|index| values[*index as usize]).collect(),
                None => values.to_vec(),
            }
        };
        let positions: Vec<Vec3> = match &kept {
            Some(kept) => kept.iter().map(|index| all[*index as usize]).collect(),
            None => all,
        };
        let elements = narrow(&all_elements);

        // A selected field wins over CPK. Without this the tree can claim an
        // object is coloured by b_factor while the render shows element
        // colours — the field is listed, so it has to actually apply.
        // A bound colour array wins over CPK. Without this the tree could claim
        // an object is coloured by b_factor while the render showed element
        // colours — an input that is bound has to actually apply.
        let tint = super::bound(bound, "colour", &store, &arrays)
            .and_then(|values| {
                super::bound_colours(values, colour, position_array.count() as usize)
            })
            .map(|colours| match &kept {
                Some(kept) => kept.iter().map(|index| colours[*index as usize]).collect(),
                None => colours,
            });
        let stick = colour.flat.to_linear().to_f32_array();
        let atom_colours: Vec<[f32; 4]> = (0..positions.len())
            .map(|index| {
                tint.as_ref().map_or_else(
                    || element_colour(elements.get(index).copied().unwrap_or(6)),
                    |colours| colours[index],
                )
            })
            .collect();

        // Nothing moved, so the vertex count is unchanged and the existing
        // buffer can simply be painted over. This is the path a colour-map drag
        // takes, and it is why the layout is cached at all.
        if !dirty.geometry {
            if let Some(layout) = layout {
                super::repaint(
                    &mut meshes,
                    mesh3d,
                    layout.colours(&atom_colours, stick, tint.is_some()),
                );
                debug!("draw: molecule repainted, {} vertices", layout.vertices());
            }
            continue;
        }

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
                atom_colours[index],
            );
        }

        let mut drawn_bonds: Vec<[u32; 2]> = Vec::new();
        if let Some(bond_array) = super::bound(bound, "bonds", &store, &arrays) {
            let pairs = bond_array.to_u32().unwrap_or_default();
            for original in pairs.chunks_exact(2) {
                // A bond is drawn only when both its atoms are: half a bond
                // sticking out into space reads as broken geometry, not as a
                // deliberate cut.
                let pair = match &remap {
                    Some(remap) => {
                        let (Some(a), Some(b)) = (remap.get(original[0]), remap.get(original[1]))
                        else {
                            continue;
                        };
                        [a, b]
                    }
                    None => [original[0], original[1]],
                };
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
                // When atoms are field-coloured, a bond takes the mean of its
                // endpoints so the mapping stays continuous along the chain.
                let bond_colour = if tint.is_some() {
                    mean(
                        atom_colours[pair[0] as usize],
                        atom_colours[pair[1] as usize],
                    )
                } else {
                    stick
                };
                merged.stamp(
                    &cylinder,
                    // Bevy's cylinder runs along +Y, so rotate that onto the bond.
                    &Transform::from_translation((*a + *b) * 0.5)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, along / length))
                        .with_scale(Vec3::new(*bond_radius, length, *bond_radius)),
                    bond_colour,
                );
                drawn_bonds.push([pair[0], pair[1]]);
            }
        }

        let vertices = merged.positions.len();
        let sticks = drawn_bonds.len();
        super::ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, merged.build());
        super::ensure_material(
            &mut commands,
            entity,
            &mut materials,
            material3d,
            StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.4,
                ..default()
            },
        );
        commands.entity(entity).insert(MoleculeLayout {
            atom_vertices: sphere.positions.len(),
            bond_vertices: cylinder.positions.len(),
            atoms: positions.len(),
            bonds: drawn_bonds,
        });

        debug!(
            "draw: molecule merged into one mesh — {} atoms, {sticks} bonds, {vertices} vertices",
            positions.len()
        );
    }
}
