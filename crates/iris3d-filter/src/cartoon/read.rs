//! Turning loose arrays into residues, and residues into runs.
//!
//! The curve is continuous through a *segment* — a run of residues with no
//! chain break and no missing guide atom. Everything here is about finding
//! those runs: reading the bound arrays into an [`Input`], grouping atoms into
//! [`Node`]s, splitting the nodes into segments, and orienting each so the
//! ribbon does not twist between one residue and the next.

use super::style::*;
use super::*;

/// The most atoms a base's outline can have: a purine's fused bicyclic.
pub(super) const RING_ATOMS: usize = 9;

/// One nucleic base, as the outline of its ring system.
///
/// This is Mol\*'s `nucleotide-ring` rather than its `nucleotide-block`. The
/// block draws every base as the same standard-sized rectangle; the ring follows
/// the actual atoms, so a purine is visibly the larger fused shape and a
/// pyrimidine a plain hexagon. It is what Mol\*'s default preset uses, and it is
/// the more honest picture — the outline is measured rather than stipulated.
#[derive(Debug, Clone, Copy)]
pub(super) struct Base {
    /// The perimeter of the ring system, in order around the outside. Only the
    /// first [`Base::corners`] entries are used.
    pub(super) perimeter: [Vec3; RING_ATOMS],
    pub(super) corners: usize,
    /// The glycosidic nitrogen, where the stick to the backbone starts.
    pub(super) attach: Vec3,
}

impl Base {
    /// Reads a base out of the atoms found for one residue, or `None` if the
    /// ring is incomplete.
    ///
    /// A purine is told from a pyrimidine by carrying the five-ring atoms at
    /// all. That is a structural test rather than a list of residue names, so a
    /// modified base with the same ring system still draws correctly and an
    /// unrecognised one is not a special case.
    ///
    /// The perimeters are the outside of each ring system, not the atom order.
    /// A purine's two rings share the `C4`-`C5` bond, so the outline goes round
    /// the six-ring to `C4`, crosses into the five-ring at `N9`, and comes back
    /// through `C5` — nine corners. A pyrimidine is the plain six.
    pub(super) fn read(found: &[Option<Vec3>; RING_ATOMS]) -> Option<Self> {
        let [n1, c2, n3, c4, c5, c6, n7, c8, n9] = *found;
        let mut perimeter = [Vec3::ZERO; RING_ATOMS];

        if let (Some(n7), Some(c8), Some(n9)) = (n7, c8, n9) {
            perimeter[..RING_ATOMS].copy_from_slice(&[n1?, c2?, n3?, c4?, n9, c8, n7, c5?, c6?]);
            return Some(Self {
                perimeter,
                corners: RING_ATOMS,
                attach: n9,
            });
        }

        perimeter[..6].copy_from_slice(&[n1?, c2?, n3?, c4?, c5?, c6?]);
        Some(Self {
            perimeter,
            corners: 6,
            attach: n1?,
        })
    }

    /// The corners of the outline, in order.
    pub(super) fn outline(&self) -> &[Vec3] {
        &self.perimeter[..self.corners]
    }

    /// The plane the ring lies in, as a centre and a unit normal.
    ///
    /// Newell's method over the whole outline rather than a cross product of
    /// three atoms: a ring is only approximately planar, and picking three of
    /// its nine atoms would let a single displaced one tilt the whole face.
    /// Newell's averages every edge and is exact for a planar polygon.
    pub(super) fn plane(&self) -> Option<(Vec3, Vec3)> {
        let outline = self.outline();
        let centre = outline.iter().copied().sum::<Vec3>() / outline.len() as f32;
        let mut normal = Vec3::ZERO;
        for corner in 0..outline.len() {
            let (from, to) = (outline[corner], outline[(corner + 1) % outline.len()]);
            normal += (from - centre).cross(to - centre);
        }
        let normal = normal.normalize_or_zero();
        (normal.length_squared() > f32::EPSILON).then_some((centre, normal))
    }
}

/// One residue's contribution to the curve.
#[derive(Debug, Clone, Copy)]
pub(super) struct Node {
    pub(super) residue: u32,
    pub(super) position: Vec3,
    /// Towards the direction atom, unnormalised. Zero when the residue had no
    /// direction atom at all, which [`orient`] fills in.
    pub(super) direction: Vec3,
    pub(super) form: Form,
    pub(super) polymer: Polymer,
    pub(super) chain: u32,
    /// The base to hang off this residue, for a nucleic one whose ring is
    /// complete. Always `None` for a protein residue.
    pub(super) base: Option<Base>,
}

/// A run of residues the curve is continuous through.
pub(super) type Segment = Vec<Node>;

/// The decoded arrays, kept alive so [`Backbone`] can borrow them.
///
/// Reading an actor's bindings into this lives here rather than in a backend
/// because it is not a pipeline decision: both pathways want exactly the same
/// six arrays, read exactly the same way. What they do with the resulting
/// [`Ribbon`] is where they part company.
pub struct Input {
    pub(super) positions: Vec<Vec3>,
    pub(super) residue_of_atom: Vec<u32>,
    pub(super) name_of_atom: Vec<u32>,
    pub(super) names: Vec<String>,
    pub(super) sse: Vec<u8>,
    pub(super) chain_of_residue: Vec<u32>,
}

impl Input {
    pub fn backbone(&self) -> Backbone<'_> {
        Backbone {
            positions: &self.positions,
            residue_of_atom: &self.residue_of_atom,
            name_of_atom: &self.name_of_atom,
            names: &self.names,
            sse: &self.sse,
            chain_of_residue: &self.chain_of_residue,
        }
    }
}

/// Reads the bound arrays, narrowed to the selected atoms.
///
/// `atoms` cuts atoms, and an atom it removes is simply not there to be a trace
/// atom — so a deselected residue breaks the curve exactly as an unresolved one
/// does. That is the honest result: splining across the hole would draw a ribbon
/// through a region the caller asked to hide.
///
/// The per-residue arrays are deliberately **not** narrowed. They are keyed on
/// the residue index, which a selection does not renumber, so cutting them would
/// misalign every residue after the first gap.
pub(super) fn read(request: &Request) -> Option<Input> {
    let positions = request.input("positions")?;
    let names = request.input("atom_name")?;
    let all_positions = positions.to_vec3();
    let all_residues = request.input("residue_index")?.to_u32()?;
    let all_names = request.input("atom_name_index")?.to_u32()?;
    if all_positions.is_empty()
        || all_residues.len() < all_positions.len()
        || all_names.len() < all_positions.len()
    {
        return None;
    }

    let kept = request.input("atoms").and_then(|array| array.to_u32());
    let narrow = |values: &[u32]| -> Vec<u32> {
        match &kept {
            Some(kept) => kept
                .iter()
                .filter_map(|index| values.get(*index as usize).copied())
                .collect(),
            None => values.to_vec(),
        }
    };

    Some(Input {
        positions: match &kept {
            Some(kept) => kept
                .iter()
                .filter_map(|index| all_positions.get(*index as usize).copied())
                .collect(),
            None => all_positions,
        },
        residue_of_atom: narrow(&all_residues),
        name_of_atom: narrow(&all_names),
        names: names.strings.clone(),
        // A `uint8` array's bytes are its values, so the codes need no decode.
        sse: request
            .input("residue_sse")
            .map(|array| array.data.clone())
            .unwrap_or_default(),
        chain_of_residue: request
            .input("residue_chain_index")
            .and_then(|array| array.to_u32())
            .unwrap_or_default(),
    })
}

/// Groups the atoms into residues, then the residues into runs the curve is
/// continuous through.
pub(super) fn segments(backbone: &Backbone) -> Vec<Segment> {
    let nodes = nodes(backbone);

    let mut segments = Vec::new();
    let mut current: Segment = Vec::new();
    for node in nodes {
        let broken = match current.last() {
            None => false,
            Some(previous) => {
                previous.chain != node.chain
                    || previous.polymer != node.polymer
                    // A residue number is not consulted. Author numbering skips
                    // and repeats, so the geometry is the honest test: two trace
                    // atoms further apart than a bond can reach are not joined,
                    // whatever they are numbered.
                    || previous.position.distance(node.position) > node.polymer.gap()
            }
        };
        if broken && !current.is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        current.push(node);
    }
    segments.push(current);

    // A single residue has no interval to interpolate over. This is also what
    // silently disposes of a calcium ion caught by the name `CA`.
    segments.retain(|segment| segment.len() >= 2);
    segments
}

/// One node per residue that has a trace atom, in residue order.
///
/// A residue with no trace atom produces nothing, which is what breaks the
/// curve there — a water, a ligand or a residue whose backbone was not resolved.
pub(super) fn nodes(backbone: &Backbone) -> Vec<Node> {
    // The dictionary is tiny — a few dozen distinct names for any structure —
    // so the roles are worked out once per name rather than once per atom.
    let roles: Vec<Option<(Polymer, Guide, u8)>> = backbone
        .names
        .iter()
        .map(|name| guide_role(name.trim()))
        .collect();

    /// Best-ranked position seen for one slot.
    #[derive(Default, Clone, Copy)]
    struct Slot(Option<(u8, Vec3)>);

    impl Slot {
        fn offer(&mut self, rank: u8, position: Vec3) {
            if self.0.is_none_or(|(held, _)| rank < held) {
                self.0 = Some((rank, position));
            }
        }
        fn position(self) -> Option<Vec3> {
            self.0.map(|(_, position)| position)
        }
    }

    #[derive(Default, Clone, Copy)]
    struct Found {
        trace: Slot,
        direction: Slot,
        polymer: Option<Polymer>,
        /// The base-ring atoms, in [`base_slot`] order.
        ring: [Option<Vec3>; RING_ATOMS],
    }

    // As for the guide roles: worked out once per distinct name rather than once
    // per atom, which is what keeps a second lookup over every atom free.
    let rings: Vec<Option<usize>> = backbone
        .names
        .iter()
        .map(|name| base_slot(name.trim()))
        .collect();

    let mut found: HashMap<u32, Found> = HashMap::default();
    let atoms = backbone
        .positions
        .len()
        .min(backbone.residue_of_atom.len())
        .min(backbone.name_of_atom.len());
    for atom in 0..atoms {
        let name = backbone.name_of_atom[atom] as usize;
        let guide = roles.get(name).copied().flatten();
        let ring = rings.get(name).copied().flatten();
        if guide.is_none() && ring.is_none() {
            continue;
        }
        let residue = found.entry(backbone.residue_of_atom[atom]).or_default();
        let position = backbone.positions[atom];

        if let Some((polymer, guide, rank)) = guide {
            match guide {
                Guide::Trace => {
                    residue.trace.offer(rank, position);
                    // The trace atom decides the polymer, so a nucleic base's
                    // `C2` cannot make a protein residue nucleic on its own.
                    residue.polymer = Some(polymer);
                }
                Guide::Direction => residue.direction.offer(rank, position),
            }
        }
        // Recorded for every residue, and read back only for a nucleic one. A
        // protein residue can carry a `C2` or an `N1` in a side chain, and
        // sorting that out here would mean knowing the residue's name; the
        // polymer test below settles it without needing to.
        if let Some(slot) = ring {
            residue.ring[slot] = Some(position);
        }
    }

    let mut nodes: Vec<Node> = found
        .into_iter()
        .filter_map(|(residue, entry)| {
            let position = entry.trace.position()?;
            let polymer = entry.polymer?;
            let form = match polymer {
                // A nucleic residue is drawn as a nucleic ribbon whatever its
                // code says. Nothing assigns secondary structure to one today,
                // so reading the column here would draw every base as coil.
                Polymer::Nucleic => Form::Nucleic,
                Polymer::Protein => backbone
                    .sse
                    .get(residue as usize)
                    .copied()
                    .map_or(Form::Coil, Form::of_code),
            };
            Some(Node {
                residue,
                position,
                direction: entry
                    .direction
                    .position()
                    .map_or(Vec3::ZERO, |atom| atom - position),
                form,
                polymer,
                chain: backbone
                    .chain_of_residue
                    .get(residue as usize)
                    .copied()
                    .unwrap_or(0),
                // Nucleic only. This is where the side-chain ambiguity noted
                // above is resolved: a protein residue never reaches `Base`,
                // whatever its atoms are called.
                base: match polymer {
                    Polymer::Nucleic => Base::read(&entry.ring),
                    Polymer::Protein => None,
                },
            })
        })
        .collect();
    // Residue order, which is the order the curve follows. The map above lost
    // it, and the atoms it was recovered from are not guaranteed to be sorted
    // either.
    nodes.sort_unstable_by_key(|node| node.residue);
    nodes
}

/// Fixes the alternating flip in the direction vectors, and fills in the ones
/// that are missing.
///
/// The flip is the whole reason this step exists. Consecutive peptide planes
/// point roughly opposite ways, so a ribbon built from the raw carbonyl vectors
/// twists 180° per residue. Negating whenever the dot product with the
/// predecessor is negative removes it, and is what Carson and Bugg did.
///
/// A residue with no direction atom inherits its predecessor's. That covers a
/// CA-only or coarse-grain model, where the alternative — a frame propagated
/// along the curve — is smoother but depends on every residue before it, and so
/// pops under animation.
pub(super) fn orient(segment: &mut Segment) {
    let mut previous = Vec3::ZERO;
    for node in segment.iter_mut() {
        if node.direction.length_squared() < f32::EPSILON {
            node.direction = previous;
            continue;
        }
        if node.direction.dot(previous) < 0.0 {
            node.direction = -node.direction;
        }
        previous = node.direction;
    }

    // A segment whose first residues had no direction atom was left with zeros,
    // because there was nothing before them to inherit. Fill backwards from the
    // first one that did.
    if let Some(first) = segment
        .iter()
        .position(|node| node.direction.length_squared() >= f32::EPSILON)
    {
        let direction = segment[first].direction;
        for node in &mut segment[..first] {
            node.direction = direction;
        }
    }
}

/// Averages the guide points of a strand with their neighbours.
///
/// A beta strand pleats: its CAs zigzag either side of the sheet by about half
/// an ångström, and a spline through them gives a ribbon that visibly ripples.
/// The weighted mean flattens it. This is NGL's smoothing rather than Mol\*'s
/// tension handling, and it applies to strands only — flattening a helix would
/// pull it onto its own axis and lose the coil.
pub(super) fn smooth_strands(segment: &mut Segment) {
    if segment.len() < 3 {
        return;
    }
    let original: Vec<Vec3> = segment.iter().map(|node| node.position).collect();
    for index in 1..segment.len() - 1 {
        if segment[index].form != Form::Strand {
            continue;
        }
        segment[index].position =
            (original[index - 1] + original[index] * 2.0 + original[index + 1]) * 0.25;
    }
}
