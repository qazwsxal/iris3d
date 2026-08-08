//! Turning uploaded arrays into structured datasets.
//!
//! **Provisional.** The wire format currently carries anonymous named arrays
//! and nothing about what they collectively describe, so structure is inferred
//! from buffer names here. That is a bootstrap, not the destination: the proto
//! needs to say what a dataset *is* — grids in particular are unreachable this
//! way, since origin, spacing and dimensions cannot be inferred from arrays.
//! When the schema grows, this module should defer to it and keep name-based
//! inference only as a fallback for the raw-buffer escape hatch.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::data::{DataArray, Field, Fields, NamedArray, NamedBuffer};
use super::dataset::{Bonds, Cells, CellKind, DatasetKind, MeshData, MoleculeData, PointCloud};

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
pub fn ingest(buffers: Vec<NamedBuffer>, assets: &mut Assets<DataArray>) -> Ingested {
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

    let (dataset, kind) = classify(&by_name);

    // Everything without a structural role is a field over the dataset.
    let mut fields = Fields::default();
    for array in &arrays {
        if ROLES.contains(&array.meta.name.as_str()) {
            continue;
        }
        fields.0.insert(
            array.meta.name.clone(),
            Field {
                kind: Field::infer_kind(&array.meta),
                // Cannot be known from the arrays alone; per-point is the
                // common case and the wire format should carry this.
                association: super::data::Association::PerPoint,
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

fn classify(by_name: &HashMap<String, NamedArray>) -> (Dataset, DatasetKind) {
    let Some(positions) = by_name.get(ROLE_POSITIONS) else {
        return (Dataset::Raw, DatasetKind::Raw);
    };

    // Molecular structure wins: an object with atoms and bonds is a molecule
    // even though it is also, technically, points joined by lines.
    let is_molecule =
        by_name.contains_key(ROLE_ELEMENTS) || by_name.contains_key(ROLE_BONDS);
    if is_molecule {
        let molecule = MoleculeData {
            positions: positions.handle.clone(),
            elements: by_name.get(ROLE_ELEMENTS).map(|a| a.handle.clone()),
            bonds: by_name.get(ROLE_BONDS).map(|pairs| Bonds {
                pairs: pairs.handle.clone(),
                orders: by_name.get(ROLE_BOND_ORDERS).map(|a| a.handle.clone()),
            }),
            // Residues and chains have no wire representation yet, so uploaded
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
