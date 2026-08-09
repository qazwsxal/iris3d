//! What an object *is*, independent of how it gets drawn.
//!
//! These are separate components rather than variants of one enum, so a system
//! that handles meshes queries `&MeshData` and never sees anything else. An
//! object carries exactly one of them (or none — raw arrays with no recognised
//! structure are still a valid scene object).
//!
//! Molecules share the primitives rather than forming a parallel hierarchy:
//! atoms are points, bonds are edges, and per-atom or per-residue quantities are
//! ordinary [`Field`](super::Field)s, so colour mapping and selection work the
//! same everywhere.

// Scaffolding: these types describe data and how it should be drawn, but no
// rendering backend consumes them yet. Scoped to this module so genuine dead
// code elsewhere still surfaces. Remove once a backend lands.
#![allow(dead_code)]

use bevy::prelude::*;
use std::ops::Range;

use super::data::DataArray;

/// An unstructured set of points. `positions` is `[n, 3]` float32.
#[derive(Component, Debug)]
pub struct PointCloud {
    pub positions: Handle<DataArray>,
}

/// Points joined by cells.
#[derive(Component, Debug)]
pub struct MeshData {
    pub positions: Handle<DataArray>,
    pub cells: Cells,
}

/// Connectivity into the position array.
#[derive(Debug)]
pub struct Cells {
    /// Indices, shaped `[m, k]` where `k` matches `kind`.
    pub connectivity: Handle<DataArray>,
    pub kind: CellKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    Lines,
    Triangles,
    Tetrahedra,
}

impl CellKind {
    /// Indices per cell.
    pub fn arity(self) -> u64 {
        match self {
            CellKind::Lines => 2,
            CellKind::Triangles => 3,
            CellKind::Tetrahedra => 4,
        }
    }
}

/// A regular axis-aligned grid. Positions are implicit in `origin` and
/// `spacing`, so no coordinate array is stored.
///
/// Not yet constructible from ingest — the wire format has no way to carry
/// origin, spacing and dimensions. See the note in `scene::ingest`.
#[derive(Component, Debug)]
pub struct GridData {
    pub origin: Vec3,
    pub spacing: Vec3,
    pub dims: UVec3,
}

impl GridData {
    pub fn point_count(&self) -> u64 {
        self.dims.x as u64 * self.dims.y as u64 * self.dims.z as u64
    }
}

/// A molecular structure: atoms, optional bonds, and the residue and chain
/// hierarchy that cartoon-style representations need.
#[derive(Component, Debug)]
pub struct MoleculeData {
    /// Atom centres, `[n, 3]` float32.
    pub positions: Handle<DataArray>,
    /// Atomic numbers, `[n]` uint8. Drives radii and element colouring.
    pub elements: Option<Handle<DataArray>>,
    pub bonds: Option<Bonds>,
    /// Empty for small molecules, which have no residue hierarchy.
    pub residues: Vec<Residue>,
    pub chains: Vec<Chain>,
}

#[derive(Debug)]
pub struct Bonds {
    /// Atom index pairs, `[m, 2]` uint32.
    pub pairs: Handle<DataArray>,
    /// Bond types, `[m]` uint8 — see [`BondType`].
    pub orders: Option<Handle<DataArray>>,
}

/// How two atoms are bonded.
///
/// The values match biotite's `BondType` enum, which iris3d adopts as its wire
/// convention rather than inventing one. Collapsing these into
/// single/double/triple/aromatic would lose the distinction between an
/// aromatic single and an aromatic double, and leave coordination bonds with
/// nowhere to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BondType {
    /// Bonded, order unspecified.
    Any = 0,
    Single = 1,
    Double = 2,
    Triple = 3,
    Quadruple = 4,
    AromaticSingle = 5,
    AromaticDouble = 6,
    AromaticTriple = 7,
    /// A metal coordination bond.
    Coordination = 8,
    /// Aromatic, with no Kekulé assignment.
    Aromatic = 9,
}

impl BondType {
    pub fn from_raw(value: u8) -> Option<Self> {
        Some(match value {
            0 => BondType::Any,
            1 => BondType::Single,
            2 => BondType::Double,
            3 => BondType::Triple,
            4 => BondType::Quadruple,
            5 => BondType::AromaticSingle,
            6 => BondType::AromaticDouble,
            7 => BondType::AromaticTriple,
            8 => BondType::Coordination,
            9 => BondType::Aromatic,
            _ => return None,
        })
    }

    /// Whether the bond is part of an aromatic system, however it was assigned.
    pub fn is_aromatic(self) -> bool {
        matches!(
            self,
            BondType::AromaticSingle
                | BondType::AromaticDouble
                | BondType::AromaticTriple
                | BondType::Aromatic
        )
    }

    /// Number of lines a bond of this type is conventionally drawn with.
    pub fn multiplicity(self) -> u8 {
        match self {
            BondType::Double | BondType::AromaticDouble => 2,
            BondType::Triple | BondType::AromaticTriple => 3,
            BondType::Quadruple => 4,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Residue {
    pub name: String,
    pub seq: i32,
    /// Atoms belonging to this residue, as a range into the position array.
    pub atoms: Range<u32>,
    pub secondary: SecondaryStructure,
}

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: String,
    /// Residues belonging to this chain, as a range into `MoleculeData::residues`.
    pub residues: Range<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecondaryStructure {
    #[default]
    Unknown,
    Coil,
    Helix,
    Sheet,
    Turn,
}

/// Which dataset component an object carries, for describing it without
/// querying each type in turn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetKind {
    Points,
    Mesh,
    Grid,
    Molecule,
    /// Arrays with no recognised structure. Still a valid object — this is the
    /// escape hatch for data iris3d does not model yet.
    Raw,
    /// No data at all. Used as a pure grouping node: children parented to it
    /// share its transform. There is no separate group type.
    Empty,
}

impl DatasetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DatasetKind::Points => "points",
            DatasetKind::Mesh => "mesh",
            DatasetKind::Grid => "grid",
            DatasetKind::Molecule => "molecule",
            DatasetKind::Raw => "raw",
            DatasetKind::Empty => "empty",
        }
    }
}
