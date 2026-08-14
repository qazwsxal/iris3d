//! Glycans as SNFG symbols, opaque, composing with the volumes around them.
//!
//! The geometry is [`crate::draw::glycan`]. What is here is the mapping onto
//! this pathway: the symbols are opaque, so they write depth, and the moment
//! accumulation truncates at that depth — a density map in front of a sugar
//! dims it and one behind does not.
//!
//! # Opaque only, and no mode to choose
//!
//! [`cartoon`](super::cartoon) offers `absorbing` because a ribbon is a plausible
//! medium: it has a thickness worth seeing through. A glycan symbol is not — it
//! is a *notation*, a stand-in for a residue rather than a depiction of one, and
//! a semi-transparent notation is just a harder-to-read notation. So there is no
//! choice here, which is the honest shape rather than a missing feature.
//!
//! Colour comes from the notation rather than from [`ColorBy`](crate::scene::ColorBy):
//! in SNFG the colour is half the identity — a blue square is GlcNAc and a
//! yellow square is GalNAc — so it is not the actor's to set.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::draw::glycan::{self, Style};
use crate::scene::link::Placement;
use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, float};
use crate::scene::{DataArray, DataStore, Dtype};

use super::{Actor, Dirty, mark};

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
        id: "residue_index",
        label: "residue per atom",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32, Dtype::Uint64],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "residue_snfg",
        label: "sugar per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "size",
        label: "symbol size (Å)",
        kind: ParamKind::Float {
            default: 1.6,
            min: 0.3,
            max: 5.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "link_radius",
        label: "link radius (Å)",
        kind: ParamKind::Float {
            default: 0.35,
            min: 0.05,
            max: 1.5,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "link_distance",
        label: "link distance (Å)",
        kind: ParamKind::Float {
            default: 7.0,
            min: 3.0,
            max: 12.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "radial_segments",
        label: "sides of a round symbol",
        kind: ParamKind::Float {
            default: 16.0,
            min: 4.0,
            max: 32.0,
            logarithmic: false,
        },
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "glycan",
        label: "glycan (SNFG)",
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(GlycanStyle(Style {
                size: float(params, "size", 1.6),
                link_radius: float(params, "link_radius", 0.35),
                link_distance: float(params, "link_distance", 7.0),
                radial_segments: float(params, "radial_segments", 16.0).round() as usize,
            }));
        },
    });
}

/// The shared [`Style`], as this backend's style component.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct GlycanStyle(pub Style);

pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<GlycanStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

type Drawable<'a> = (
    Actor<'a, GlycanStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<StandardMaterial>>,
);

pub fn draw_glycans(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, _colour, subset, bindings, dirty), mesh3d, material3d) in &dirty {
        if !dirty.geometry {
            continue;
        }
        let Some(input) = glycan::read(bindings, subset, &store, &arrays) else {
            continue;
        };
        let (symbols, colours) = glycan::build(&input, &style.0);
        if symbols.is_empty() {
            debug!("draw: a glycan actor found no sugars; nothing to draw");
            continue;
        }

        debug!(
            "draw: opaque glycan rebuilt — {} vertices, {} triangles",
            symbols.positions.len(),
            symbols.indices.len() / 3
        );

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, symbols.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, symbols.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
        mesh.insert_indices(Indices::U32(symbols.indices));

        if let Some(Mesh3d(handle)) = mesh3d
            && let Some(mut slot) = meshes.get_mut(handle)
        {
            *slot = mesh;
        } else {
            commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
        }

        if material3d.is_none() {
            commands
                .entity(entity)
                .insert(MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 0.9,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                })));
        }
    }
}

/// Gives every placement of a glycan the mesh and material it holds.
///
/// As for [`cartoon::place_cartoons`](super::cartoon::place_cartoons): these
/// actors carry no `MomentVolume`, so the pathway's own placement system does
/// not match them.
#[allow(clippy::type_complexity)]
pub fn place_glycans(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<GlycanStyle>>,
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
