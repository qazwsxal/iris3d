//! Triangle meshes, handed to Bevy's standard PBR pipeline.
//!
//! Only `CellKind::Triangles` is drawn. Tetrahedra are skipped deliberately:
//! drawing a volumetric mesh as a surface means extracting its boundary faces
//! first (collect every face, drop the ones shared by two cells), which is a
//! separate piece of work rather than something to bodge in here. Lines are
//! skipped for the same reason — they want a line representation, not a
//! surface.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::data::Fields;
use crate::scene::dataset::CellKind;
use crate::scene::{ColorBy, DataArray, MeshData, Representation};

use super::NeedsRedraw;

pub fn draw_surfaces(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    dirty: Query<(Entity, &Representation, &ColorBy, &ChildOf), With<NeedsRedraw>>,
    surfaces: Query<(&MeshData, Option<&Fields>)>,
) {
    for (entity, representation, colour, parent) in &dirty {
        if !matches!(representation, Representation::Surface) {
            continue;
        }
        let Ok((data, fields)) = surfaces.get(parent.parent()) else {
            continue;
        };
        if data.cells.kind != CellKind::Triangles {
            info!(
                "draw: nothing drawn for {:?} cells — a surface needs boundary \
                 extraction first",
                data.cells.kind
            );
            continue;
        }

        let (Some(position_array), Some(index_array)) = (
            arrays.get(&data.positions),
            arrays.get(&data.cells.connectivity),
        ) else {
            continue;
        };
        let positions = position_array.to_vec3();
        let Some(indices) = index_array.to_u32() else {
            warn!("draw: mesh indices are not an integer type");
            continue;
        };
        if positions.is_empty() || indices.is_empty() {
            continue;
        }
        if let Some(out_of_range) = indices.iter().find(|i| **i as usize >= positions.len()) {
            warn!(
                "draw: mesh index {out_of_range} exceeds {} vertices",
                positions.len()
            );
            continue;
        }

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            positions.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
        );
        mesh.insert_indices(Indices::U32(indices));

        // "normals" has no structural role in ingest, so it arrives as an
        // ordinary three-component field. Use it when it is there, otherwise
        // derive flat normals so the surface is at least shaded.
        let supplied = fields
            .and_then(|fields| fields.0.get("normals"))
            .and_then(|field| arrays.get(&field.array))
            .map(|array| array.to_vec3())
            .filter(|normals| normals.len() == positions.len());

        match supplied {
            Some(normals) => mesh.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                normals.iter().map(|n| [n.x, n.y, n.z]).collect::<Vec<_>>(),
            ),
            None => mesh.compute_normals(),
        }

        let tinted = super::colour_field(colour, fields)
            .and_then(|field| super::vertex_colours(field, colour, &arrays, positions.len()));
        let tinted = match tinted {
            Some(colours) => {
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
                true
            }
            None => false,
        };

        let vertices = positions.len();
        commands.entity(entity).insert((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                // Vertex colours multiply the base, so it has to be white for
                // them to come through unaltered.
                base_color: if tinted { Color::WHITE } else { colour.flat },
                perceptual_roughness: 0.55,
                double_sided: true,
                cull_mode: None,
                ..default()
            })),
        ));

        debug!("draw: surface with {vertices} vertices");
    }
}
