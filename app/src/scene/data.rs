//! Arrays and the meaning attached to them.
//!
//! Raw bytes live in [`DataArray`], a Bevy asset, so several representations of
//! one object share a single copy and get change detection for free. Everything
//! that says what the bytes *mean* — element type, shape, whether a field is
//! scalar, vector or tensor, whether it sits on points or cells — is kept
//! outside the array, because the same bytes legitimately mean different things
//! in different contexts.

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
    // Unused until subsets, which validate a selection's length against the
    // element count of the array it selects into.
    #[allow(dead_code)]
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
    // As for `BufferMeta::count`: wanted by subset validation.
    #[allow(dead_code)]
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
            Dtype::Uint64 => decode!(u64, 8),
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
    /// Never constructed yet: `ingest` hardcodes `PerPoint` because the wire
    /// format cannot say otherwise. Kept because subsets reuse this enum to say
    /// which domain a selection indexes into.
    #[allow(dead_code)]
    PerCell,
}

/// A named quantity defined over a dataset.
#[derive(Debug, Clone)]
pub struct Field {
    pub kind: FieldKind,
    /// Unread until a backend distinguishes per-cell from per-point values;
    /// `vertex_colours` currently assumes per-point for everything.
    #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    const DTYPES: [Dtype; 10] = [
        Dtype::Uint8,
        Dtype::Int8,
        Dtype::Uint16,
        Dtype::Int16,
        Dtype::Uint32,
        Dtype::Int32,
        Dtype::Uint64,
        Dtype::Int64,
        Dtype::Float32,
        Dtype::Float64,
    ];

    /// Whether `to_u32` is defined for this type at all.
    fn integral(dtype: Dtype) -> bool {
        !matches!(dtype, Dtype::Float32 | Dtype::Float64)
    }

    /// Encodes `values` as little-endian bytes of `dtype`. Callers keep values
    /// inside `i8` range so that every type — the narrowest signed one
    /// included — represents them exactly and every decode should agree.
    fn array(dtype: Dtype, values: &[u64]) -> DataArray {
        let mut data = Vec::new();
        for value in values {
            match dtype {
                Dtype::Uint8 | Dtype::Int8 => data.push(*value as u8),
                Dtype::Uint16 => data.extend((*value as u16).to_le_bytes()),
                Dtype::Int16 => data.extend((*value as i16).to_le_bytes()),
                Dtype::Uint32 => data.extend((*value as u32).to_le_bytes()),
                Dtype::Int32 => data.extend((*value as i32).to_le_bytes()),
                Dtype::Uint64 => data.extend(value.to_le_bytes()),
                Dtype::Int64 => data.extend((*value as i64).to_le_bytes()),
                Dtype::Float32 => data.extend((*value as f32).to_le_bytes()),
                Dtype::Float64 => data.extend((*value as f64).to_le_bytes()),
            }
        }
        DataArray {
            dtype,
            shape: vec![values.len() as u64],
            data,
        }
    }

    /// Every dtype decodes to the same numbers. `to_u32` on a `Uint64` array
    /// used to decode with a four-byte stride and panic on the `try_into`,
    /// which meant any client uploading numpy's default integer width crashed
    /// the application rather than seeing a rejected upload.
    #[test]
    fn every_dtype_round_trips() {
        let values = [0u64, 1, 7, 42, 127];
        for dtype in DTYPES {
            let decoded = array(dtype, &values);
            assert_eq!(
                decoded.to_f32(),
                values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                "to_f32 disagreed for {dtype}"
            );
            if integral(dtype) {
                assert_eq!(
                    decoded.to_u32(),
                    Some(values.iter().map(|v| *v as u32).collect::<Vec<_>>()),
                    "to_u32 disagreed for {dtype}"
                );
            }
        }
    }

    /// Byte length matters as much as value: a decode that reads the wrong
    /// stride can still return the right count for a short input.
    #[test]
    fn decodes_the_full_array_at_every_width() {
        let values: Vec<u64> = (0..64).collect();
        for dtype in DTYPES {
            let decoded = array(dtype, &values);
            assert_eq!(
                decoded.data.len() as u64,
                dtype.size() * values.len() as u64,
                "encoder wrote the wrong width for {dtype}"
            );
            assert_eq!(decoded.to_f32().len(), values.len(), "to_f32 for {dtype}");
            if integral(dtype) {
                assert_eq!(
                    decoded.to_u32().map(|v| v.len()),
                    Some(values.len()),
                    "to_u32 for {dtype}"
                );
            }
        }
    }

    #[test]
    fn floats_have_no_index_reading() {
        for dtype in [Dtype::Float32, Dtype::Float64] {
            assert!(array(dtype, &[1, 2, 3]).to_u32().is_none());
        }
    }

    #[test]
    fn signed_types_survive_negative_values() {
        let bytes: Vec<u8> = [-1i32, -128, 0, 127]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = DataArray {
            dtype: Dtype::Int32,
            shape: vec![4],
            data: bytes,
        };
        assert_eq!(array.to_f32(), vec![-1.0, -128.0, 0.0, 127.0]);
    }

    #[test]
    fn to_vec3_needs_three_components() {
        let three = DataArray {
            dtype: Dtype::Float32,
            shape: vec![2, 3],
            data: (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect(),
        };
        assert_eq!(
            three.to_vec3(),
            vec![Vec3::new(0.0, 1.0, 2.0), Vec3::new(3.0, 4.0, 5.0)]
        );

        let two = DataArray {
            dtype: Dtype::Float32,
            shape: vec![3, 2],
            data: (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect(),
        };
        assert!(two.to_vec3().is_empty());
    }

    #[test]
    fn byte_length_reports_the_packed_size() {
        let meta = BufferMeta {
            name: "positions".into(),
            dtype: Dtype::Float32,
            shape: vec![10, 3],
        };
        assert_eq!(meta.byte_length(), Some(120));
        assert_eq!(meta.count(), 10);
        assert_eq!(meta.components(), 3);

        // A shape whose product overflows must not wrap around into a small
        // allocation the upload path would happily accept.
        let huge = BufferMeta {
            name: "huge".into(),
            dtype: Dtype::Float64,
            shape: vec![u64::MAX, 2],
        };
        assert_eq!(huge.byte_length(), None);
    }
}
