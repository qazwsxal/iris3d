//! Renumbering what survives when elements are dropped.
//!
//! # What used to be here, and why it is not
//!
//! This module was a `Subset` **component on an actor**: a selection arrived
//! inline on `AddActor`, was held beside the bindings, and three kinds —
//! `points`, `ball-and-stick` and `glycan` — narrowed their own arrays with it
//! while drawing.
//!
//! That made an actor decide *what* to draw, which is not an actor's job. An
//! actor is a dumb consumer of arrays: it reads what it is handed and puts it on
//! screen. Choosing which elements to hand it is a question about data, and
//! questions about data belong to filters. It is the same category error the
//! `cartoon` actor made before it became a filter, one level down.
//!
//! It also could not be *wired*. A selection carried inline on the actor is not
//! a handle, so it appears nowhere in the graph, cannot be shared between two
//! actors, and cannot be computed from the data it selects over.
//!
//! **Subsetting is now three filters**, in [`filter::index`](crate::filter::index):
//!
//! - `gather` narrows per-element data — positions, elements, B-factors;
//! - `renumber` narrows connectivity, which is what [`Remap`] below is for;
//! - `reindex` re-densifies a hierarchy index whose numbering went sparse.
//!
//! and the selection itself comes from `subset`, over a mask built by `compare`,
//! `logic` and `match`. An actor sees only arrays that are already narrowed and
//! is unchanged by any of it.
//!
//! # What stayed
//!
//! [`Remap`] alone, because the rule it implements is unchanged by the move: an
//! entry of a connectivity array survives only when *every* element it names
//! does. That is VTK's extract-selection rule, it was right when an actor
//! applied it, and it is right now that `renumber` does. The tests below are the
//! ones that pinned it, kept verbatim.

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
