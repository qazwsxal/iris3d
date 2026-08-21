//! The numbers, and what describes them.
//!
//! - The `array` module — element types, an array and its metadata, and the
//!   [`DataStore`] that holds every array and mesh a client has uploaded or a
//!   filter has produced, flat and by handle.
//! - [`chem`] — facts about the periodic table: an element's radius and its
//!   conventional colour.
//!
//! Chemistry sits beside the arrays rather than under the renderer because a
//! filter tinting by element wants the same answer a renderer does, and a
//! periodic table under `draw` would make every filter that mentions an element
//! depend on how pixels are produced.
//!
//! Nothing here knows what a scene is, what a filter is, or how anything is
//! drawn. An array belongs to no object: that is the point of holding them flat,
//! and it is what lets one array feed several representations.

pub mod array;
pub mod chem;

pub use array::{BufferMeta, DataArray, DataStore, Dtype, HeldMeta, NamedBuffer};
