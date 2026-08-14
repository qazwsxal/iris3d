//! Triangle meshes, raytraced.
//!
//! The one kind here that is not instanced: a surface is a single mesh already,
//! so it keeps the default backend's shape of one `Mesh3d` and one material per
//! actor. What differs is where the colour comes from. Solari cannot read
//! vertex colours, so the scalar is written into `UV.x` and a 1D ramp texture
//! supplies the colour — see [`super::ramp_uvs`].

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, flag};
use crate::scene::subset::Remap;
use crate::scene::{DataArray, DataStore, Dtype};

use super::{Actor, Dirty, RampPalette, bound, condition, mark, normalised, ramp_uvs};

/// Cell surfaces, shaded.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct MeshStyle {
    /// Light and draw back faces as well as front ones. On by default because
    /// scientific meshes are routinely open or inconsistently wound, and a
    /// one-sided material renders those as holes.
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
    ParamSpec {
        id: "indices",
        label: "triangles",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 3],
            required: true,
        },
    },
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
        id: "mesh",
        label: "mesh",
        // A triangle mesh is one mesh, rasterised or raytraced.
        shared: true,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(MeshStyle {
                double_sided: flag(params, "double_sided", true),
            });
        },
    });
}

/// `double_sided` is a material property; nothing about the mesh depends on it.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<MeshStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
}

/// What this backend needs to redraw one surface: everything any backend needs,
/// plus what it produced last time. Rebuilding into the assets the actor already
/// holds is what keeps a slider drag from allocating.
type Drawable<'a> = (
    Actor<'a, MeshStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<StandardMaterial>>,
);

#[allow(clippy::too_many_arguments)]
pub fn draw_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut palette: ResMut<RampPalette>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, colour, subset, bindings, dirty), mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }
        let (Some(position_array), Some(index_array)) = (
            bound(bindings, "positions", &store, &arrays),
            bound(bindings, "indices", &store, &arrays),
        ) else {
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

        // A triangle survives only if all three corners do, and the survivors
        // are renumbered, so connectivity is rewritten rather than filtered.
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

        let ramp = bound(bindings, "colour", &store, &arrays)
            .and_then(|values| normalised(values, colour, position_array.count() as usize))
            .map(|ramp| match &kept {
                Some(kept) => kept.iter().map(|index| ramp[*index as usize]).collect(),
                None => ramp,
            });

        if dirty.geometry || dirty.colour {
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

            let supplied = bound(bindings, "normals", &store, &arrays)
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

            // One vertex per element here, so the mapping is the identity. A
            // kind that expanded elements into several vertices each would pass
            // the expansion instead.
            if let Some(ramp) = &ramp {
                ramp_uvs(&mut mesh, ramp, |vertex| vertex);
            }
            // Last: tangents need the UVs and the normals to be in place.
            condition(&mut mesh);

            match mesh3d {
                Some(Mesh3d(handle)) if meshes.get(handle).is_some() => {
                    if let Some(mut slot) = meshes.get_mut(handle) {
                        *slot = mesh;
                    }
                }
                _ => {
                    commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
                }
            }
            debug!("draw: solari surface, {} vertices", positions.len());
        }

        // Whether a ramp is bound changes the material, not only the vertices,
        // so this runs on any dirt at all.
        let base_colour_texture = ramp
            .is_some()
            .then(|| palette.image(colour.map, &mut images));
        let material = StandardMaterial {
            // The texture multiplies the base, so it has to be white for the
            // ramp to come through unaltered.
            base_color: if base_colour_texture.is_some() {
                Color::WHITE
            } else {
                colour.flat
            },
            base_color_texture: base_colour_texture,
            perceptual_roughness: 0.55,
            double_sided: style.double_sided,
            cull_mode: if style.double_sided {
                None
            } else {
                Some(bevy::render::render_resource::Face::Back)
            },
            ..default()
        };
        match material3d {
            Some(MeshMaterial3d(handle)) if materials.get(handle).is_some() => {
                if let Some(mut slot) = materials.get_mut(handle) {
                    *slot = material;
                }
            }
            _ => {
                commands
                    .entity(entity)
                    .insert(MeshMaterial3d(materials.add(material)));
            }
        }
    }
}
