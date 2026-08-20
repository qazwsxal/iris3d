//! Renumbering what survives when elements are dropped.
//!
//! Dropping elements renumbers everything after them, and a mesh's cells and a
//! molecule's bonds refer to points by index — so their references have to be
//! rewritten to match, not merely filtered. [`Remap`] is that rewrite.
//!
//! Narrowing itself is not here. It is three filters in
//! [`filter::index`](crate::filter::index), which is where the decision about
//! *what* to draw belongs — see `docs/design/filters.md`. This module holds only
//! the renumbering rule they share.

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
