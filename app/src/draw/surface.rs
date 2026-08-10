//! Triangle meshes, handed to Bevy's standard PBR pipeline.
//!
//! Triangles only: the `indices` input declares `[n, 3]`, so a tetrahedral or
//! line connectivity array cannot be bound to it at all, and the caller is told
//! why. Drawing a volumetric mesh as a surface means extracting its boundary
//! faces first, which is a separate piece of work rather than something to bodge
//! in here; lines want a line actor.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, flag};
use crate::scene::subset::Remap;
use crate::scene::{DataArray, DataStore, Dtype};

use super::{Dirty, Drawable, mark};

/// Cell surfaces, shaded.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct SurfaceStyle {
    /// Light and draw back faces as well as front ones.
    ///
    /// On by default because scientific meshes are routinely open or have
    /// inconsistent winding, and a one-sided material renders those as holes.
    /// Turning it off is how you see through to the inside of a closed mesh.
    pub double_sided: bool,
}

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "positions",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
        },
    },
    // Triangles only, and the shape says so. Tetrahedra used to arrive and be
    // refused at draw time with a log line nobody reads; an `[n, 4]` array now
    // simply cannot be bound here, and the reason comes back from the call that
    // tried.
    ParamSpec {
        id: "indices",
        label: "triangles",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 3],
            required: true,
        },
    },
    // Unbound means "work them out from the triangles", which is what happened
    // when an upload carried none.
    ParamSpec {
        id: "normals",
        label: "normals",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable<SurfaceStyle, StandardMaterial>>,
) {
    for (entity, style, colour, subset, bound, _source, dirty, mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let (Some(position_array), Some(index_array)) = (
            super::bound(bound, "positions", &store, &arrays),
            super::bound(bound, "indices", &store, &arrays),
        ) else {
            // Both are required, so `check_bindings` refused an actor without
            // them; reaching here means an array was released underneath one.
            continue;
        };
        let all = position_array.to_vec3();
        let Some(all_indices) = index_array.to_u32() else {
            warn!("draw: mesh indices are not an integer type");
            continue;
        };
        if all.is_empty() || all_indices.is_empty() {
            continue;
        }
        if let Some(out_of_range) = all_indices.iter().find(|i| **i as usize >= all.len()) {
            warn!(
                "draw: mesh index {out_of_range} exceeds {} vertices",
                all.len()
            );
            continue;
        }

        // A triangle survives only if all three of its corners do, and the
        // surviving points are renumbered, so the connectivity has to be
        // rewritten rather than merely filtered.
        let kept = subset.selected(all.len(), &arrays);
        let (positions, indices) = match &kept {
            Some(kept) => {
                let remap = Remap::new(kept, all.len());
                let positions: Vec<Vec3> = kept.iter().map(|index| all[*index as usize]).collect();
                let indices: Vec<u32> = all_indices
                    .chunks_exact(3)
                    .filter_map(|corners| remap.cell(corners))
                    .flatten()
                    .collect();
                if indices.is_empty() {
                    info!("draw: a subset left no whole triangles; nothing to draw");
                    continue;
                }
                (positions, indices)
            }
            None => (all, all_indices),
        };

        let tint = super::bound(bound, "colour", &store, &arrays)
            .and_then(|values| {
                super::bound_colours(values, colour, position_array.count() as usize)
            })
            .map(|colours| match &kept {
                Some(kept) => kept.iter().map(|index| colours[*index as usize]).collect(),
                None => colours,
            });

        if dirty.geometry {
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                positions
                    .iter()
                    .map(|p| [p.x, p.y, p.z])
                    .collect::<Vec<_>>(),
            );
            mesh.insert_indices(Indices::U32(indices));

            // Use the bound normals when there are some, otherwise derive flat
            // ones so the surface is at least shaded.
            let supplied = super::bound(bound, "normals", &store, &arrays)
                .map(|array| array.to_vec3())
                .filter(|normals| normals.len() == position_array.count() as usize)
                .map(|normals| match &kept {
                    Some(kept) => kept.iter().map(|index| normals[*index as usize]).collect(),
                    None => normals,
                });

            match supplied {
                Some(normals) => mesh.insert_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    normals.iter().map(|n| [n.x, n.y, n.z]).collect::<Vec<_>>(),
                ),
                None => mesh.compute_normals(),
            }

            if let Some(colours) = tint.clone() {
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
            }
            super::ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, mesh);
            debug!("draw: surface rebuilt with {} vertices", positions.len());
        } else if dirty.colour
            && let Some(colours) = tint.clone()
        {
            super::repaint(&mut meshes, mesh3d, colours);
        }

        // Colouring is a material change too, not only a vertex one: whether the
        // base is white or the flat colour depends on there being a tint at all,
        // so switching a field on or off has to rewrite both.
        if dirty.any() {
            super::ensure_material(
                &mut commands,
                entity,
                &mut materials,
                material3d,
                StandardMaterial {
                    // Vertex colours multiply the base, so it has to be white for
                    // them to come through unaltered.
                    base_color: if tint.is_some() {
                        Color::WHITE
                    } else {
                        colour.flat
                    },
                    perceptual_roughness: 0.55,
                    double_sided: style.double_sided,
                    // `double_sided` only lights the back faces; they still have
                    // to survive culling to be lit at all.
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
}
