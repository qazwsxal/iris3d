//! Arrays and meshes, and nothing about what they mean.
//!
//! Raw bytes live in [`DataArray`], a Bevy asset, so several actors binding the
//! same array share one copy and get change detection for free. [`DataStore`]
//! holds them by the handle a client knows them by.
//!
//! What the bytes *mean* is not recorded here, and deliberately not recorded at
//! all: it is decided when an array is bound to an actor's input. The same
//! numbers can be positions for one actor and a colour ramp for another, so
//! there was never one right answer to store.
//!
//! # Geometry is the other thing a handle can name
//!
//! A handle names either an array or **one mesh** — vertices, triangles and the
//! attributes on them, as a single `Handle<Mesh>`. Two kinds of thing in one
//! handle space, so no id ever names both and a client asking what a handle is
//! gets one answer.
//!
//! Geometry exists because arrays could not express the thing that mattered.
//! Positions, indices and normals as three arrays are three *descriptions* of a
//! mesh, and every consumer that wanted to draw them had to assemble its own —
//! so a ribbon drawn as a lit surface and as an absorbing medium put
//! the same vertices on the GPU twice. One `Handle<Mesh>` is one upload, referenced by
//! however many actors want it. See [`crate::filter::geometry`], which is what
//! turns arrays into one.

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
    /// Text, one string per element.
    ///
    /// The one type whose elements are not raw bytes: the strings live in
    /// [`DataArray::strings`] and the byte buffer stays empty. It exists
    /// because labelling data — a chain's id, a residue's name — is not
    /// numeric and had nowhere to go.
    Str,
}

impl Dtype {
    /// Size of a single element in bytes.
    ///
    /// Zero for [`Dtype::Str`], which is exactly right rather than a stand-in:
    /// a string array's byte buffer really is empty, so every length it
    /// computes — declared size, expected chunk total — comes out at 0 and the
    /// upload path needs no special case to let it through.
    pub fn size(self) -> u64 {
        match self {
            Dtype::Str => 0,
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
            Dtype::Str => "string",
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
    // Wanted once a client can ask what shape a held array is.
    #[allow(dead_code)]
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
    /// Populated instead of `data` when the meta says [`Dtype::Str`]. Empty
    /// otherwise.
    pub strings: Vec<String>,
}

/// Everything about a mesh except its vertices.
///
/// The counterpart of [`BufferMeta`], and thin for the same reason: enough to
/// tell a client what a handle names and enough for an input to decide whether
/// it can read it, without being a second copy of the mesh itself.
///
/// `normals` and `colours` are here rather than being looked up on the asset
/// because both decide something outside the render world — a `medium` with no
/// shell never asks for normals, and a `surface` uses its flat `tint` exactly when
/// the geometry carries no colours — and neither should have to reach into
/// `Assets<Mesh>` to find out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeometryMeta {
    pub name: String,
    pub vertices: u64,
    pub triangles: u64,
    /// A normal per vertex. Read by lighting and by a glass shell; never by the
    /// absorbance accumulation, which cares where a boundary is and not which
    /// way it faces.
    pub normals: bool,
    /// A **linear RGB** colour per vertex, already mapped. Which ramp and what
    /// range produced them was the `colormap` filter's business.
    pub colours: bool,
}

/// Every array and mesh a client holds, by handle.
///
/// Getting numbers into the scene and putting a node in the tree are two
/// operations, not one: an array may feed several representations, and a
/// representation may read arrays that arrived at different times. So arrays are
/// held here, flat, and an actor names the ones it wants.
///
/// The store owns a strong `Handle`, which is what keeps the asset alive. Drop
/// the entry and the bytes go once nothing else still refers to them — an actor
/// holding the same handle keeps it loaded, which is why releasing is described
/// as forgetting rather than freeing.
///
/// Two maps, one handle space. Ids come from the same sequence, so a handle
/// names an array or a mesh and never both, and [`held`](Self::held) is the
/// answer to "what is this?" for a caller that does not already know.
#[derive(Resource, Default)]
pub struct DataStore {
    arrays: HashMap<u64, StoredArray>,
    geometry: HashMap<u64, StoredGeometry>,
    /// Data handle to the filter that writes it, for the handles a filter
    /// allocated as its outputs.
    ///
    /// Held here rather than derived from the filter entities, so that "may I
    /// forget this array" is a question the store answers on its own. Releasing
    /// an array something is still generating would leave that filter producing
    /// into nothing, which looks exactly like a filter that has broken.
    generated: HashMap<u64, u64>,
}

/// One held array: what it is, and where its bytes live.
pub struct StoredArray {
    pub meta: BufferMeta,
    /// Never read — held for what dropping it does. This is the strong
    /// reference that keeps the asset loaded, so the store's `HashMap` entry is
    /// the array's lifetime.
    #[allow(dead_code)]
    pub handle: Handle<DataArray>,
}

/// One held mesh. As [`StoredArray`], and the handle is read here rather than
/// merely held: an actor drawing this geometry clones it into a `Mesh3d`.
pub struct StoredGeometry {
    pub meta: GeometryMeta,
    pub handle: Handle<Mesh>,
}

/// What a handle names, described but not held.
///
/// For the callers that have to cope with either — a binding check, a listing,
/// an input picker — rather than for the many that want one particular sort and
/// would rather be told `None`. The description rather than the asset, so
/// something that has copied a listing out of the store can still ask an input
/// whether it would accept it.
#[derive(Debug, Clone, Copy)]
pub enum Held<'a> {
    Array(&'a BufferMeta),
    Geometry(&'a GeometryMeta),
}

impl<'a> Held<'a> {
    pub fn name(self) -> &'a str {
        match self {
            Held::Array(meta) => &meta.name,
            Held::Geometry(meta) => &meta.name,
        }
    }
}

/// [`Held`], owned. What crosses the wire and what the interface keeps a copy
/// of.
#[derive(Debug, Clone)]
pub enum HeldMeta {
    Array(BufferMeta),
    Geometry(GeometryMeta),
}

impl HeldMeta {
    pub fn as_held(&self) -> Held<'_> {
        match self {
            HeldMeta::Array(meta) => Held::Array(meta),
            HeldMeta::Geometry(meta) => Held::Geometry(meta),
        }
    }

    pub fn name(&self) -> &str {
        self.as_held().name()
    }

    /// One line naming what this is, for a listing or a picker: an array's type
    /// and shape, a mesh's size.
    pub fn describe(&self) -> String {
        match self {
            HeldMeta::Array(meta) => format!("{}{:?}", meta.dtype, meta.shape),
            HeldMeta::Geometry(meta) => {
                format!("mesh · {} verts, {} tris", meta.vertices, meta.triangles)
            }
        }
    }
}

impl DataStore {
    pub fn insert(&mut self, id: u64, meta: BufferMeta, handle: Handle<DataArray>) {
        self.arrays.insert(id, StoredArray { meta, handle });
    }

    pub fn insert_geometry(&mut self, id: u64, meta: GeometryMeta, handle: Handle<Mesh>) {
        self.geometry.insert(id, StoredGeometry { meta, handle });
    }

    /// The array under a handle, or `None` — which covers a released handle and
    /// a handle that names geometry rather than numbers.
    ///
    /// Named for what it returns rather than being a bare `get`, because the
    /// store holds two things and a caller asking the wrong one deserves to see
    /// it in the call.
    pub fn array(&self, id: u64) -> Option<&StoredArray> {
        self.arrays.get(&id)
    }

    /// The mesh under a handle. See [`array`](Self::array).
    pub fn geometry(&self, id: u64) -> Option<&StoredGeometry> {
        self.geometry.get(&id)
    }

    /// Whatever the handle names, for a caller that accepts either.
    pub fn held(&self, id: u64) -> Option<Held<'_>> {
        self.arrays
            .get(&id)
            .map(|array| Held::Array(&array.meta))
            .or_else(|| {
                self.geometry
                    .get(&id)
                    .map(|mesh| Held::Geometry(&mesh.meta))
            })
    }

    /// Forgets an array or a mesh, reporting whether it was held at all.
    /// Records that `filter` writes `data`, when a filter's outputs are
    /// allocated. Undone by [`forget_generated`](Self::forget_generated).
    pub fn mark_generated(&mut self, data: u64, filter: u64) {
        self.generated.insert(data, filter);
    }

    /// Drops the ownership record for one of a filter's outputs.
    pub fn forget_generated(&mut self, data: u64) {
        self.generated.remove(&data);
    }

    /// The filter that writes this handle, if any.
    pub fn generated_by(&self, data: u64) -> Option<u64> {
        self.generated.get(&data).copied()
    }

    pub fn remove(&mut self, id: u64) -> bool {
        self.arrays.remove(&id).is_some() | self.geometry.remove(&id).is_some()
    }

    /// Every array held, in handle order so a listing is stable between calls.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &StoredArray)> {
        let mut ids: Vec<u64> = self.arrays.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| self.arrays.get(&id).map(|array| (id, array)))
    }

    /// Every mesh held. See [`iter`](Self::iter).
    pub fn iter_geometry(&self) -> impl Iterator<Item = (u64, &StoredGeometry)> {
        let mut ids: Vec<u64> = self.geometry.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .filter_map(|id| self.geometry.get(&id).map(|mesh| (id, mesh)))
    }
}

/// Raw array bytes, shared by handle.
///
/// `Clone` for one caller: a filter runs on a worker thread, which cannot borrow
/// from the world, so its inputs are copied in. That copy is a real cost — a
/// 256³ float grid is 64 MB — and is why cloning one of these deliberately looks
/// like work rather than happening implicitly. Everything else shares the
/// `Handle`.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct DataArray {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
    /// The elements of a [`Dtype::Str`] array, densely packed in the same C
    /// order `data` uses. Empty for every other type.
    ///
    /// A second buffer rather than an enum over the two: `data` is read by
    /// every backend and reshaping it into a variant would touch all of them
    /// to express something no renderer asks about. Nothing draws a string.
    ///
    /// Read by the cartoon builder, which needs the distinct atom
    /// names to tell a `CA` from an `O`. That is the shape this was built for:
    /// the geometry is decided by a text column, and the text arrives once per
    /// distinct value rather than once per atom.
    pub strings: Vec<String>,
}

impl DataArray {
    /// A numeric array, with no strings.
    ///
    /// Most callers build one of these, so the empty `strings` is worth not
    /// repeating.
    pub fn numeric(dtype: Dtype, shape: Vec<u64>, data: Vec<u8>) -> Self {
        Self {
            dtype,
            shape,
            data,
            strings: Vec::new(),
        }
    }

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
            // Text has no numeric reading. Empty rather than zeros: a caller
            // colour-mapping this should see no values, not a flat field of
            // them that looks like real data.
            Dtype::Str => Vec::new(),
        }
    }

    /// Decodes the bytes as `u32` indices, widening narrower integer types.
    /// Returns `None` for floating-point and text arrays, which have no index
    /// reading.
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
            Dtype::Float32 | Dtype::Float64 | Dtype::Str => None,
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
                // Text is not built from numbers. The string array tests
                // construct their own.
                Dtype::Str => unreachable!("DTYPES holds no text"),
            }
        }
        DataArray::numeric(dtype, vec![values.len() as u64], data)
    }

    /// Every dtype decodes to the same numbers, whatever its width.
    ///
    /// The trap is a stride that does not match the dtype — decoding a `Uint64`
    /// four bytes at a time panics on the `try_into`, so a client uploading
    /// numpy's default integer width would crash the application rather than
    /// see a rejected upload.
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
        let array = DataArray::numeric(Dtype::Int32, vec![4], bytes);
        assert_eq!(array.to_f32(), vec![-1.0, -128.0, 0.0, 127.0]);
    }

    #[test]
    fn to_vec3_needs_three_components() {
        let three = DataArray::numeric(
            Dtype::Float32,
            vec![2, 3],
            (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect(),
        );
        assert_eq!(
            three.to_vec3(),
            vec![Vec3::new(0.0, 1.0, 2.0), Vec3::new(3.0, 4.0, 5.0)]
        );

        let two = DataArray::numeric(
            Dtype::Float32,
            vec![3, 2],
            (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect(),
        );
        assert!(two.to_vec3().is_empty());
    }

    /// A string array occupies no bytes, so the length the upload path expects
    /// it to receive is zero and it is complete the moment it is declared. This
    /// falls out of `Dtype::Str::size()` being 0 rather than being special-cased
    /// anywhere, which is the whole reason the size is 0.
    #[test]
    fn text_declares_no_bytes_at_any_length() {
        for count in [1u64, 3, 100_000] {
            let meta = BufferMeta {
                name: "res_name".into(),
                dtype: Dtype::Str,
                shape: vec![count],
            };
            assert_eq!(meta.byte_length(), Some(0), "for {count} strings");
            assert_eq!(meta.count(), count);
        }
    }

    /// Text has no numeric reading, and asking for one gives nothing rather
    /// than a plausible-looking field of zeros — a colour map handed zeros
    /// would paint a flat surface that looks like real data.
    #[test]
    fn text_has_no_numeric_reading() {
        let array = DataArray {
            dtype: Dtype::Str,
            shape: vec![3],
            data: Vec::new(),
            strings: vec!["ALA".into(), "GLY".into(), "HOH".into()],
        };
        assert!(array.to_f32().is_empty());
        assert_eq!(array.to_u32(), None);
        assert!(array.to_vec3().is_empty());
        assert_eq!(array.strings.len(), 3);
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
