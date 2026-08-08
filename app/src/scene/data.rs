//! Arrays and the meaning attached to them.
//!
//! Raw bytes live in [`DataArray`], a Bevy asset, so several representations of
//! one object share a single copy and get change detection for free. Everything
//! that says what the bytes *mean* — element type, shape, whether a field is
//! scalar, vector or tensor, whether it sits on points or cells — is kept
//! outside the array, because the same bytes legitimately mean different things
//! in different contexts.

// Scaffolding: these types describe data and how it should be drawn, but no
// rendering backend consumes them yet. Scoped to this module so genuine dead
// code elsewhere still surfaces. Remove once a backend lands.
#![allow(dead_code)]

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use std::fmt::{self, Display};

/// Element type of an array's raw bytes. Always little-endian, densely packed
/// in C (row-major) order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,
    Float32,
    Float64,
}

impl Dtype {
    /// Size of a single element in bytes.
    pub fn size(self) -> u64 {
        match self {
            Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 => 8,
        }
    }
}

impl Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dtype::Uint8 => "uint8",
            Dtype::Int8 => "int8",
            Dtype::Uint16 => "uint16",
            Dtype::Int16 => "int16",
            Dtype::Uint32 => "uint32",
            Dtype::Int32 => "int32",
            Dtype::Uint64 => "uint64",
            Dtype::Int64 => "int64",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
        };
        f.write_str(name)
    }
}

/// Everything about an array except its contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferMeta {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

impl BufferMeta {
    /// Bytes a densely packed array of this shape and type occupies, or `None`
    /// if the product overflows.
    pub fn byte_length(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.size(), |acc, axis| acc.checked_mul(*axis))
    }

    /// Length of the outermost axis — the number of points, cells or atoms.
    pub fn count(&self) -> u64 {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Number of components per element: `[n, 3]` has three, `[n]` has one.
    pub fn components(&self) -> u64 {
        self.shape.iter().skip(1).product()
    }
}

/// A named array plus its contents, as delivered by ingest and before it
/// becomes an asset.
#[derive(Debug)]
pub struct NamedBuffer {
    pub meta: BufferMeta,
    pub data: Vec<u8>,
}

/// Raw array bytes, shared by handle.
#[derive(Asset, TypePath, Debug)]
pub struct DataArray {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl DataArray {
    pub fn count(&self) -> u64 {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Number of components per element: `[n, 3]` has three, `[n]` has one.
    pub fn components(&self) -> u64 {
        self.shape.iter().skip(1).product()
    }

    /// Decodes the bytes as `f32`, converting from other numeric types.
    ///
    /// Decoding element by element rather than casting the slice: the bytes
    /// arrive in a `Vec<u8>` with no alignment guarantee, so a zero-copy cast
    /// to `&[f32]` would be unsound. It also keeps the little-endian contract
    /// from the wire format explicit rather than assuming host order.
    pub fn to_f32(&self) -> Vec<f32> {
        macro_rules! decode {
            ($ty:ty, $width:expr) => {
                self.data
                    .chunks_exact($width)
                    .map(|bytes| <$ty>::from_le_bytes(bytes.try_into().unwrap()) as f32)
                    .collect()
            };
        }
        match self.dtype {
            Dtype::Float32 => decode!(f32, 4),
            Dtype::Float64 => decode!(f64, 8),
            Dtype::Uint8 => self.data.iter().map(|b| *b as f32).collect(),
            Dtype::Int8 => self.data.iter().map(|b| *b as i8 as f32).collect(),
            Dtype::Uint16 => decode!(u16, 2),
            Dtype::Int16 => decode!(i16, 2),
            Dtype::Uint32 => decode!(u32, 4),
            Dtype::Int32 => decode!(i32, 4),
            Dtype::Uint64 => decode!(u64, 8),
            Dtype::Int64 => decode!(i64, 8),
        }
    }

    /// Decodes the bytes as `u32` indices, widening narrower integer types.
    /// Returns `None` for floating-point arrays.
    pub fn to_u32(&self) -> Option<Vec<u32>> {
        macro_rules! decode {
            ($ty:ty, $width:expr) => {
                Some(
                    self.data
                        .chunks_exact($width)
                        .map(|bytes| <$ty>::from_le_bytes(bytes.try_into().unwrap()) as u32)
                        .collect(),
                )
            };
        }
        match self.dtype {
            Dtype::Uint8 => Some(self.data.iter().map(|b| *b as u32).collect()),
            Dtype::Uint16 => decode!(u16, 2),
            Dtype::Uint32 => decode!(u32, 4),
            Dtype::Uint64 => decode!(u64, 4),
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => {
                Some(self.to_f32().into_iter().map(|v| v as u32).collect())
            }
            Dtype::Float32 | Dtype::Float64 => None,
        }
    }

    /// Decodes an `[n, 3]` array as positions. Returns an empty vector if the
    /// array is not three-component.
    pub fn to_vec3(&self) -> Vec<Vec3> {
        if self.components() != 3 {
            return Vec::new();
        }
        self.to_f32()
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect()
    }
}

/// An array belonging to a scene object, retained so objects can be described
/// without dereferencing their contents.
#[derive(Debug, Clone)]
pub struct NamedArray {
    pub meta: BufferMeta,
    pub handle: Handle<DataArray>,
}

/// What a field's components represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// One component per element.
    Scalar,
    /// Three components per element.
    Vector,
    /// A rank-2 tensor — stress, strain, diffusion.
    Tensor(TensorLayout),
}

/// How a tensor's components are packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// All nine components, row-major.
    Full9,
    /// The six unique components of a symmetric tensor, in Voigt order:
    /// `xx, yy, zz, yz, xz, xy`. The usual storage for stress and strain.
    SymmetricVoigt6,
}

/// What a field's values are attached to.
///
/// The distinction the raw arrays cannot express: identical bytes mean
/// different things depending on whether each value belongs to a point or to
/// a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Association {
    PerPoint,
    PerCell,
}

/// A named quantity defined over a dataset.
#[derive(Debug, Clone)]
pub struct Field {
    pub kind: FieldKind,
    pub association: Association,
    pub array: Handle<DataArray>,
    pub meta: BufferMeta,
}

/// The fields defined over an object, keyed by name.
#[derive(Component, Debug, Default)]
pub struct Fields(pub HashMap<String, Field>);

impl Field {
    /// Infers a field kind from an array's component count.
    ///
    /// Provisional: six components are read as symmetric Voigt tensors, which
    /// is the common convention for stress and strain but is a guess. Once the
    /// wire format carries field kind explicitly this should defer to it.
    pub fn infer_kind(meta: &BufferMeta) -> FieldKind {
        match meta.components() {
            9 => FieldKind::Tensor(TensorLayout::Full9),
            6 => FieldKind::Tensor(TensorLayout::SymmetricVoigt6),
            3 => FieldKind::Vector,
            _ => FieldKind::Scalar,
        }
    }
}
