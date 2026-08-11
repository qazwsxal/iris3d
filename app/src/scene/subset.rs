//! Drawing part of a dataset rather than all of it.
//!
//! This is what several actors of one object are *for*: chain A as cartoon and
//! the ligand as ball-and-stick is one structure drawn twice, each time over a
//! different part of it. Without a subset the second actor can only be the
//! whole thing drawn again.
//!
//! Selections arrive as index or mask arrays, not as a query language. iris3d
//! already takes the position that clients parse their own formats and push
//! arrays — see the note at the top of `scene.proto` — and a selection language
//! is a parser. Python has the residue and chain structure to hand and can
//! express "chain A" far more naturally than a string grammar invented here
//! would; what reaches the scene is the answer, not the question.

use bevy::prelude::*;

use super::data::{Association, DataArray, Dtype};

/// A selection as it arrives from outside the ECS, before its values become a
/// shared asset.
///
/// Mirrors how uploads work: the transport has no access to the world, so raw
/// bytes cross the channel and the scene turns them into an asset on its own
/// tick.
#[derive(Debug, Clone)]
pub struct SubsetRequest {
    pub data: Vec<u8>,
    pub dtype: Dtype,
    pub encoding: SubsetEncoding,
    pub association: Association,
}

impl SubsetRequest {
    /// Number of values, given the element width.
    pub fn count(&self) -> u64 {
        self.data.len() as u64 / self.dtype.size()
    }

    pub fn into_subset(self, arrays: &mut Assets<DataArray>) -> Subset {
        Subset::Selected {
            array: arrays.add(DataArray {
                dtype: self.dtype,
                shape: vec![self.count()],
                data: self.data,
            }),
            encoding: self.encoding,
            association: self.association,
        }
    }
}

/// How a selection array says which elements it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsetEncoding {
    /// Element indices, in any order. Out-of-range entries are dropped.
    Indices,
    /// One value per element, non-zero to keep.
    Mask,
}

/// Which part of the source dataset an actor draws.
#[derive(Component, Debug, Clone, Default)]
pub enum Subset {
    #[default]
    All,
    Selected {
        array: Handle<DataArray>,
        encoding: SubsetEncoding,
        /// Which domain the selection indexes — points and atoms, or cells and
        /// bonds. Reuses the field association enum because it is the same
        /// distinction, and having one of them is what stops per-cell subsets
        /// being a later breaking change.
        association: Association,
    },
}

impl Subset {
    /// Element indices to keep, or `None` for all of them.
    ///
    /// `None` is the fast path and means exactly "no filtering", so a kind
    /// can skip the remap entirely rather than build the identity permutation.
    ///
    /// A selection that does not fit the data — the wrong length, or nothing
    /// left after dropping out-of-range indices — falls back to keeping
    /// everything with a warning. Drawing nothing would look identical to a
    /// broken renderer, and a stale selection outliving a reupload is the
    /// obvious way to get here.
    pub fn selected(&self, count: usize, arrays: &Assets<DataArray>) -> Option<Vec<u32>> {
        let Subset::Selected {
            array,
            encoding,
            association: _,
        } = self
        else {
            return None;
        };
        let Some(array) = arrays.get(array) else {
            warn!("draw: a subset's selection array is missing; drawing everything");
            return None;
        };

        let kept: Vec<u32> = match encoding {
            SubsetEncoding::Indices => {
                let Some(indices) = array.to_u32() else {
                    warn!("draw: a subset's indices are a floating-point type; drawing everything");
                    return None;
                };
                indices
                    .into_iter()
                    .filter(|index| (*index as usize) < count)
                    .collect()
            }
            SubsetEncoding::Mask => {
                let values = array.to_f32();
                if values.len() != count {
                    warn!(
                        "draw: a subset mask has {} entries for {count} elements; \
                         drawing everything",
                        values.len()
                    );
                    return None;
                }
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, keep)| **keep != 0.0)
                    .map(|(index, _)| index as u32)
                    .collect()
            }
        };

        if kept.is_empty() {
            warn!("draw: a subset selects nothing; drawing everything");
            return None;
        }
        if kept.len() == count {
            // Selecting the whole thing is the same as not selecting, and
            // taking the fast path avoids a pointless remap.
            return None;
        }
        Some(kept)
    }
}

/// How many elements a selection names, without validating it against any
/// particular dataset.
///
/// Reported rather than the selection itself: the caller sent the values and
/// they can be large, so echoing them back would make every listing carry a
/// copy of every mask in the scene.
pub fn size(subset: &Subset, arrays: &Assets<DataArray>) -> Option<u64> {
    let Subset::Selected {
        array, encoding, ..
    } = subset
    else {
        return None;
    };
    let array = arrays.get(array)?;
    Some(match encoding {
        SubsetEncoding::Indices => array.count(),
        SubsetEncoding::Mask => array.to_f32().iter().filter(|keep| **keep != 0.0).count() as u64,
    })
}

/// Maps original element index to its position in the compacted output.
///
/// Needed because dropping elements renumbers everything after them: a mesh's
/// cells and a molecule's bonds refer to points by index, so their references
/// have to be rewritten to match, not merely filtered.
pub struct Remap {
    /// `to[original] = Some(compacted)`, `None` for a dropped element.
    to: Vec<Option<u32>>,
}

impl Remap {
    pub fn new(kept: &[u32], count: usize) -> Self {
        let mut to = vec![None; count];
        for (compacted, original) in kept.iter().enumerate() {
            if let Some(slot) = to.get_mut(*original as usize) {
                *slot = Some(compacted as u32);
            }
        }
        Self { to }
    }

    pub fn get(&self, original: u32) -> Option<u32> {
        self.to.get(original as usize).copied().flatten()
    }

    /// Whether every one of a cell's corners survived.
    ///
    /// A cell is kept only when all of its points are, following VTK's
    /// extract-selection: keeping a triangle with a dropped corner would mean
    /// inventing a position for it, and clamping to a surviving neighbour draws
    /// a stretched sliver across the cut rather than a clean boundary.
    pub fn cell(&self, corners: &[u32]) -> Option<Vec<u32>> {
        corners.iter().map(|corner| self.get(*corner)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::data::Dtype;

    fn array(dtype: Dtype, data: Vec<u8>, len: usize) -> DataArray {
        DataArray {
            dtype,
            shape: vec![len as u64],
            data,
        }
    }

    fn indices(assets: &mut Assets<DataArray>, values: &[u32]) -> Subset {
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Subset::Selected {
            array: assets.add(array(Dtype::Uint32, data, values.len())),
            encoding: SubsetEncoding::Indices,
            association: Association::PerPoint,
        }
    }

    fn mask(assets: &mut Assets<DataArray>, values: &[bool]) -> Subset {
        let data = values.iter().map(|v| *v as u8).collect();
        Subset::Selected {
            array: assets.add(array(Dtype::Uint8, data, values.len())),
            encoding: SubsetEncoding::Mask,
            association: Association::PerPoint,
        }
    }

    #[test]
    fn all_selects_nothing_in_particular() {
        let assets = Assets::<DataArray>::default();
        assert!(Subset::All.selected(10, &assets).is_none());
    }

    #[test]
    fn indices_are_kept_in_the_order_given() {
        let mut assets = Assets::<DataArray>::default();
        let subset = indices(&mut assets, &[3, 1]);
        assert_eq!(subset.selected(10, &assets), Some(vec![3, 1]));
    }

    #[test]
    fn a_mask_keeps_the_non_zero_entries() {
        let mut assets = Assets::<DataArray>::default();
        let subset = mask(&mut assets, &[true, false, true, false]);
        assert_eq!(subset.selected(4, &assets), Some(vec![0, 2]));
    }

    /// A selection left over from a larger upload must not index off the end of
    /// the new data.
    #[test]
    fn out_of_range_indices_are_dropped() {
        let mut assets = Assets::<DataArray>::default();
        let subset = indices(&mut assets, &[0, 99, 2]);
        assert_eq!(subset.selected(3, &assets), Some(vec![0, 2]));
    }

    /// Both degenerate cases draw everything rather than nothing: an empty
    /// render is indistinguishable from a broken one.
    #[test]
    fn a_selection_that_fits_nothing_falls_back_to_everything() {
        let mut assets = Assets::<DataArray>::default();
        let empty = indices(&mut assets, &[50, 60]);
        assert!(empty.selected(3, &assets).is_none());

        let wrong_length = mask(&mut assets, &[true, true]);
        assert!(wrong_length.selected(9, &assets).is_none());
    }

    #[test]
    fn selecting_everything_takes_the_fast_path() {
        let mut assets = Assets::<DataArray>::default();
        let subset = indices(&mut assets, &[0, 1, 2]);
        assert!(
            subset.selected(3, &assets).is_none(),
            "a full selection should not cost a remap"
        );
    }

    #[test]
    fn remapping_renumbers_what_survives() {
        let remap = Remap::new(&[1, 3], 5);
        assert_eq!(remap.get(1), Some(0));
        assert_eq!(remap.get(3), Some(1));
        assert_eq!(remap.get(0), None);
        assert_eq!(remap.get(99), None);
    }

    #[test]
    fn a_cell_needs_every_corner() {
        let remap = Remap::new(&[0, 1, 2], 5);
        assert_eq!(remap.cell(&[0, 1, 2]), Some(vec![0, 1, 2]));
        // Corner 4 was dropped, so the triangle goes with it.
        assert_eq!(remap.cell(&[0, 1, 4]), None);
    }
}
