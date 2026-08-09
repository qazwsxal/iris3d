//! Turning uploaded arrays into structured datasets.
//!
//! Structure is inferred from buffer names, with one exception: a grid is
//! *declared*, because there is nothing to infer it from. A grid's sample
//! positions are implicit, so no array reveals its presence and none carries
//! its spacing.
//!
//! A declaration always wins over inference. The rest of the recognition is
//! still name-based, which remains a bootstrap rather than the destination —
//! the wire format should eventually let a client state any dataset kind
//! outright, with names kept only as a fallback for the raw-buffer escape
//! hatch.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::data::{Association, DataArray, Field, Fields, NamedArray, NamedBuffer};
use super::dataset::{
    Bonds, CellKind, Cells, DatasetKind, GridData, MeshData, MoleculeData, PointCloud,
};

/// Buffer names with a structural role. Anything else becomes a field.
const ROLE_POSITIONS: &str = "positions";
const ROLE_INDICES: &str = "indices";
const ROLE_ELEMENTS: &str = "elements";
const ROLE_BONDS: &str = "bonds";
const ROLE_BOND_ORDERS: &str = "bond_orders";

const ROLES: [&str; 5] = [
    ROLE_POSITIONS,
    ROLE_INDICES,
    ROLE_ELEMENTS,
    ROLE_BONDS,
    ROLE_BOND_ORDERS,
];

/// The dataset component to attach, kept as one return type so the caller can
/// insert exactly one.
pub enum Dataset {
    Points(PointCloud),
    Mesh(MeshData),
    Grid(GridData),
    Molecule(MoleculeData),
    /// No recognised structure. The object still exists and still holds its
    /// arrays; it simply has nothing to draw yet.
    Raw,
}

pub struct Ingested {
    pub arrays: Vec<NamedArray>,
    pub dataset: Dataset,
    pub kind: DatasetKind,
    pub fields: Fields,
}

/// Moves uploaded buffers into the asset store and classifies them.
///
/// `grid` is the client's declaration that these buffers are fields sampled
/// over a regular grid. It wins over name-based inference, because it is the
/// one thing the names cannot tell us.
pub fn ingest(
    buffers: Vec<NamedBuffer>,
    grid: Option<GridData>,
    assets: &mut Assets<DataArray>,
) -> Ingested {
    let mut arrays = Vec::with_capacity(buffers.len());
    let mut by_name: HashMap<String, NamedArray> = HashMap::new();

    for buffer in buffers {
        let handle = assets.add(DataArray {
            dtype: buffer.meta.dtype,
            shape: buffer.meta.shape.clone(),
            data: buffer.data,
        });
        let array = NamedArray {
            meta: buffer.meta,
            handle,
        };
        by_name.insert(array.meta.name.clone(), array.clone());
        arrays.push(array);
    }

    let (dataset, kind) = match grid {
        Some(grid) => (Dataset::Grid(grid), DatasetKind::Grid),
        None => classify(&by_name),
    };

    // Everything without a structural role is a field over the dataset. A grid
    // has no structural arrays at all, so for one of those this is every array.
    let mut fields = Fields::default();
    for array in &arrays {
        if grid.is_none() && ROLES.contains(&array.meta.name.as_str()) {
            continue;
        }
        fields.0.insert(
            array.meta.name.clone(),
            Field {
                kind: Field::infer_kind(&array.meta),
                association: association(array, grid),
                array: array.handle.clone(),
                meta: array.meta.clone(),
            },
        );
    }

    Ingested {
        arrays,
        dataset,
        kind,
        fields,
    }
}

/// Whether a field's values sit on points or on cells.
///
/// Only a grid can answer this, and only because it is the one upload where the
/// server knows both counts without being told: the sample count and the cell
/// count both follow from `dims`. Everywhere else the arrays are silent on the
/// question, so per-point stands as the common case until the wire format
/// carries it.
fn association(array: &NamedArray, grid: Option<GridData>) -> Association {
    let Some(grid) = grid else {
        return Association::PerPoint;
    };
    let count = array.meta.count();
    if count == grid.cell_count() && count != grid.point_count() {
        return Association::PerCell;
    }
    if count != grid.point_count() {
        warn!(
            "ingest: field \"{}\" has {count} values, which is neither the grid's \
             {} samples nor its {} cells; treating it as per-point",
            array.meta.name,
            grid.point_count(),
            grid.cell_count(),
        );
    }
    Association::PerPoint
}

fn classify(by_name: &HashMap<String, NamedArray>) -> (Dataset, DatasetKind) {
    let Some(positions) = by_name.get(ROLE_POSITIONS) else {
        return (Dataset::Raw, DatasetKind::Raw);
    };

    // Molecular structure wins: an object with atoms and bonds is a molecule
    // even though it is also, technically, points joined by lines.
    let is_molecule = by_name.contains_key(ROLE_ELEMENTS) || by_name.contains_key(ROLE_BONDS);
    if is_molecule {
        let molecule = MoleculeData {
            positions: positions.handle.clone(),
            elements: by_name.get(ROLE_ELEMENTS).map(|a| a.handle.clone()),
            bonds: by_name.get(ROLE_BONDS).map(|pairs| Bonds {
                pairs: pairs.handle.clone(),
                orders: by_name.get(ROLE_BOND_ORDERS).map(|a| a.handle.clone()),
            }),
            // Residues and chains have no wire actor yet, so uploaded
            // molecules are currently flat atom sets. Cartoon rendering will
            // need these before it can work.
            residues: Vec::new(),
            chains: Vec::new(),
        };
        return (Dataset::Molecule(molecule), DatasetKind::Molecule);
    }

    if let Some(indices) = by_name.get(ROLE_INDICES) {
        let kind = match indices.meta.components() {
            2 => Some(CellKind::Lines),
            3 => Some(CellKind::Triangles),
            4 => Some(CellKind::Tetrahedra),
            _ => None,
        };
        if let Some(cell_kind) = kind {
            let mesh = MeshData {
                positions: positions.handle.clone(),
                cells: Cells {
                    connectivity: indices.handle.clone(),
                    kind: cell_kind,
                },
            };
            return (Dataset::Mesh(mesh), DatasetKind::Mesh);
        }
        warn!(
            "ingest: \"indices\" has {} components, which is not a cell arity; \
             treating the object as a point cloud",
            indices.meta.components()
        );
    }

    (
        Dataset::Points(PointCloud {
            positions: positions.handle.clone(),
        }),
        DatasetKind::Points,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::data::{BufferMeta, Dtype};

    fn buffer(name: &str, shape: Vec<u64>) -> NamedBuffer {
        let elements: u64 = shape.iter().product();
        NamedBuffer {
            meta: BufferMeta {
                name: name.into(),
                dtype: Dtype::Float32,
                shape,
            },
            data: vec![0; elements as usize * 4],
        }
    }

    fn grid(dims: [u32; 3]) -> GridData {
        GridData {
            origin: Vec3::ZERO,
            spacing: Vec3::ONE,
            dims: UVec3::from(dims),
        }
    }

    fn ingested(buffers: Vec<NamedBuffer>, grid: Option<GridData>) -> Ingested {
        let mut assets = Assets::<DataArray>::default();
        ingest(buffers, grid, &mut assets)
    }

    #[test]
    fn a_declared_grid_is_a_grid() {
        let result = ingested(vec![buffer("density", vec![8])], Some(grid([2, 2, 2])));
        assert_eq!(result.kind, DatasetKind::Grid);
        assert!(matches!(result.dataset, Dataset::Grid(_)));
    }

    /// Without the declaration the same buffer is unrecognised, which is the
    /// whole reason the wire format has to carry it.
    #[test]
    fn the_same_buffers_without_a_declaration_are_raw() {
        let result = ingested(vec![buffer("density", vec![8])], None);
        assert_eq!(result.kind, DatasetKind::Raw);
    }

    /// A grid carries no geometry, so every buffer is a field — including one
    /// whose name would be structural on any other dataset.
    #[test]
    fn every_buffer_of_a_grid_is_a_field() {
        let result = ingested(
            vec![buffer("density", vec![8]), buffer("normals", vec![8, 3])],
            Some(grid([2, 2, 2])),
        );
        assert_eq!(result.fields.0.len(), 2);
        assert!(result.fields.0.contains_key("density"));
    }

    /// The one upload where the server can tell point data from cell data,
    /// because `dims` gives it both counts.
    #[test]
    fn association_follows_the_element_count() {
        let result = ingested(
            vec![
                // 3x3x3 = 27 samples, 2x2x2 = 8 cells.
                buffer("at_samples", vec![27]),
                buffer("at_cells", vec![8]),
            ],
            Some(grid([3, 3, 3])),
        );
        assert_eq!(
            result.fields.0["at_samples"].association,
            Association::PerPoint
        );
        assert_eq!(
            result.fields.0["at_cells"].association,
            Association::PerCell
        );
    }

    /// A field matching neither count is still kept — the data is the client's
    /// to explain — but it must not be silently called per-cell.
    #[test]
    fn a_field_of_the_wrong_length_falls_back_to_per_point() {
        let result = ingested(vec![buffer("odd", vec![5])], Some(grid([3, 3, 3])));
        assert_eq!(result.fields.0["odd"].association, Association::PerPoint);
    }

    /// A single-sample axis spans no cells, so a flat slice has no cell count
    /// to confuse a field with.
    #[test]
    fn a_flat_grid_has_samples_but_no_cells() {
        let flat = grid([4, 4, 1]);
        assert_eq!(flat.point_count(), 16);
        assert_eq!(flat.cell_count(), 0);

        let result = ingested(vec![buffer("slice", vec![16])], Some(flat));
        assert_eq!(result.fields.0["slice"].association, Association::PerPoint);
    }

    #[test]
    fn inference_still_works_when_no_grid_is_declared() {
        let points = ingested(vec![buffer("positions", vec![4, 3])], None);
        assert_eq!(points.kind, DatasetKind::Points);

        let mesh = ingested(
            vec![
                buffer("positions", vec![4, 3]),
                buffer("indices", vec![2, 3]),
            ],
            None,
        );
        assert_eq!(mesh.kind, DatasetKind::Mesh);
    }
}
