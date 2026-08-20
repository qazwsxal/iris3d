//! Cartoon ribbons: the curve through a biopolymer backbone, swept into a solid.
//!
//! Atoms in, triangles out. Where a ribbon *goes* is a fact about the backbone,
//! as an element's radius is a fact about the periodic table, so this decides
//! nothing about how the triangles reach the screen.
//!
//! # This was an actor kind, and that was the mistake
//!
//! The curve and the sweep always lived apart from the drawing — the code below
//! is nearly unchanged — but its only caller was one actor's draw system. The
//! ribbon existed for one frame, inside one actor, and nothing else could see
//! it. So showing a ribbon as an *absorbing medium* rather than a lit surface meant
//! giving that kind a `mode` parameter, which duplicated the whole difference
//! between the `surface` and `medium` actor kinds inside a third
//! place that had no business knowing about either.
//!
//! As a filter there is no `mode`. The triangles are arrays; bind them to `surface`
//! and the ribbon is lit, bind them to `medium` and it is a medium you see
//! through, bind them to both and it is both — built once. Adding a third way of
//! displaying triangles will not touch this file.
//!
//! # The construction
//!
//! Carson and Bugg's, 1986, which is what every viewer still does:
//!
//! 1. One guide point per residue, from the trace atom — `CA` in a protein, `P`
//!    in a nucleic acid.
//! 2. One direction per residue, from the trace atom towards a second atom that
//!    fixes which way the flat ribbon faces. In a protein that is the carbonyl
//!    `O`, so the ribbon lies in the peptide plane.
//! 3. Consecutive peptide planes alternate by roughly 180°, so every direction
//!    whose dot product with its predecessor is negative gets negated. Skip this
//!    and the ribbon twists once per residue.
//! 4. A cardinal spline through the guide points, sampled
//!    [`Style::linear_segments`] times per residue, with a frame at each sample.
//! 5. A closed cross-section swept along the frames — an ellipse for a helix or
//!    a coil, a rectangle for a strand, plus an arrowhead.
//!
//! # Ported from Mol\*
//!
//! The interpolation and the profile set follow Mol\*'s `polymer-trace-mesh`
//! and `curve-segment`: the tension pinned to 0.5 at a secondary-structure
//! boundary and raised inside one, the three-point averaging pass over the
//! frames, the `aspect_ratio` and `arrow_factor` controls, and the swapped frame
//! for nucleic acids. Mol\* is:
//!
//! > Copyright (c) 2018-2024 Mol\* contributors, licensed under MIT.
//!
//! Two things differ, both forced by the moment pathway in
//! `default`.
//! Every run is **closed and capped**, because a moment pathway reads an open
//! mesh as a solid whose far wall is infinitely distant and draws it too clear;
//! Mol\*'s two-segment ribbon has no closed form and is not offered. And a run
//! whose width steps — an arrowhead's back face — is swept as its own capped
//! piece rather than as part of its neighbour. Abutting closed solids cost
//! nothing there, because absorbance is additive and an interior wall is not
//! double counted.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use iris3d_data::array::Dtype;
use iris3d_model::{ParamKind, ParamSpec, flag, float};
use iris3d_scene::DataArray;

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "atom centres",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "residue_index",
        label: "residue per atom",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32, Dtype::Uint64],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    // The two halves of a dictionary-encoded name column, which is how text
    // travels: once per distinct value, never once per atom.
    ParamSpec {
        id: "atom_name_index",
        label: "atom name per atom",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "atom_name",
        label: "distinct atom names",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Str],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "residue_sse",
        label: "secondary structure per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    ParamSpec {
        id: "residue_chain_index",
        label: "chain per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32, Dtype::Uint64],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    // Which atoms to build from, by index. Unbound uses all of them.
    //
    // Narrowing happens here rather than on the consumer: cutting the
    // *vertices* of a finished ribbon would cut triangles apart, and "draw
    // chain A" means rebuilding the curve from fewer atoms rather than hiding
    // some of it.
    ParamSpec {
        id: "atoms",
        label: "atoms to build from",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0],
            required: false,
            structural: true,
        },
    },
    ParamSpec {
        id: "size_factor",
        label: "half thickness (Å)",
        kind: ParamKind::Float {
            default: 0.2,
            min: 0.02,
            max: 1.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "aspect_ratio",
        label: "width / thickness",
        kind: ParamKind::Float {
            default: 5.0,
            min: 1.0,
            max: 15.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "nucleic_aspect_ratio",
        label: "nucleic width / thickness",
        kind: ParamKind::Float {
            default: 8.0,
            min: 1.0,
            max: 20.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "arrow_factor",
        label: "arrowhead width",
        kind: ParamKind::Float {
            default: 1.5,
            min: 1.0,
            max: 3.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "linear_segments",
        label: "samples per residue",
        kind: ParamKind::Float {
            default: 8.0,
            min: 2.0,
            max: 24.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "radial_segments",
        label: "sides of a round profile",
        kind: ParamKind::Float {
            default: 16.0,
            min: 3.0,
            max: 32.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "tubular_helices",
        label: "helices as tubes",
        kind: ParamKind::Bool { default: false },
    },
    ParamSpec {
        id: "base_rings",
        label: "nucleic base rings",
        kind: ParamKind::Bool { default: true },
    },
];

/// Arrays, not one assembled mesh.
///
/// Deliberate, though this filter could build a [`Mesh`]
/// itself. A ribbon is coloured by sending `residue_index` through `colormap`,
/// and those colours have to reach the *vertex buffer* — so the assembly has to
/// happen after the colouring, which means after this. See
/// [`geometry`](super::geometry), which is where the arrays become one mesh.
///
/// A filter whose output nothing needs to colour, such as a contour of a field
/// it has already sampled, can and should produce geometry directly.
const OUTPUTS: &[OutputSpec] = &[
    OutputSpec {
        id: "positions",
        label: "positions",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Float32),
            shape: &[0, 3],
        },
        // A vertex belongs to a residue, and `residue_index` below is the map
        // saying which. That array exists for colouring; it is also exactly
        // what a pick walks back along to name an atom.
        provenance: Provenance::Map {
            via: "residue_index",
            of: "residue_index",
        },
    },
    OutputSpec {
        id: "indices",
        label: "triangles",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[0, 3],
        },
        // Triangles, not elements of anything upstream.
        provenance: Provenance::Opaque,
    },
    OutputSpec {
        id: "normals",
        label: "normals",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Float32),
            shape: &[0, 3],
        },
        provenance: Provenance::Map {
            via: "residue_index",
            of: "residue_index",
        },
    },
    // Per *vertex*, not per residue. This is what makes the ribbon colourable
    // without a cartoon-specific colour path: send it through `colormap` for the
    // N-to-C rainbow, or through a gather for anything else keyed on residue.
    //
    // The old actor kind carried the same mapping as a private `CartoonLayout`
    // component and expanded per-residue colours onto vertices itself. Emitting
    // it is what lets that code go.
    OutputSpec {
        id: "residue_index",
        label: "residue per vertex",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Uint32),
            shape: &[0],
        },
        // The map itself, so it is its own provenance.
        provenance: Provenance::Map {
            via: "residue_index",
            of: "residue_index",
        },
    },
];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "cartoon",
        label: "cartoon ribbon",
        params: PARAMS,
        outputs: OUTPUTS,
        run: Some(run),
    });
}

fn run(request: &Request) -> Outcome {
    let mut products = Products::new();
    let Some(input) = read(request) else {
        return Outcome::refused(
            "could not read its atoms: one of positions, residue_index, \
             atom_name_index or atom_name is unbound or the wrong length",
        );
    };

    let style = Style {
        size_factor: float(&request.params, "size_factor", 0.2),
        aspect_ratio: float(&request.params, "aspect_ratio", 5.0),
        nucleic_aspect_ratio: float(&request.params, "nucleic_aspect_ratio", 8.0),
        arrow_factor: float(&request.params, "arrow_factor", 1.5),
        linear_segments: float(&request.params, "linear_segments", 8.0).round() as usize,
        radial_segments: float(&request.params, "radial_segments", 16.0).round() as usize,
        tubular_helices: flag(&request.params, "tubular_helices", false),
        base_rings: flag(&request.params, "base_rings", true),
    };

    let ribbon = build(&input.backbone(), &style);
    if ribbon.is_empty() {
        // No backbone to follow. Producing nothing leaves whatever was there
        // before, which is right: an input that says nothing teaches nothing.
        // Worth *saying*, though — a ribbon that draws nothing for a ligand-only
        // selection is correct, and a ribbon that draws nothing because the
        // atom names were bound to the wrong array is not, and they look
        // identical on screen.
        return Outcome::refused("found no protein or nucleic backbone in the atoms it was given");
    }

    let vertices = ribbon.positions.len() as u64;
    products.insert(
        "positions",
        DataArray::numeric(Dtype::Float32, vec![vertices, 3], floats(&ribbon.positions)).into(),
    );
    products.insert(
        "normals",
        DataArray::numeric(Dtype::Float32, vec![vertices, 3], floats(&ribbon.normals)).into(),
    );
    products.insert(
        "indices",
        DataArray::numeric(
            Dtype::Uint32,
            vec![ribbon.indices.len() as u64 / 3, 3],
            ribbon
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect(),
        )
        .into(),
    );
    products.insert(
        "residue_index",
        DataArray::numeric(
            Dtype::Uint32,
            vec![vertices],
            ribbon
                .residue
                .iter()
                .flat_map(|r| r.to_le_bytes())
                .collect(),
        )
        .into(),
    );
    products.into()
}

/// Triples of `f32` as little-endian bytes, which is what the wire format is.
fn floats(values: &[[f32; 3]]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|triple| triple.iter().flat_map(|v| v.to_le_bytes()))
        .collect()
}

/// How wide, how thick and how finely a cartoon is drawn.
///
/// Defaults are Mol\*'s, and [`size_factor`](Self::size_factor) is read here as
/// the half-thickness in ångströms so that the pair of it and
/// [`aspect_ratio`](Self::aspect_ratio) lands on the conventional cartoon: 0.4 Å
/// thick and 2.0 Å wide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style {
    /// Half the thickness of a ribbon, in ångströms. Also a coil's radius.
    pub size_factor: f32,
    /// How many times wider than thick a protein ribbon is.
    pub aspect_ratio: f32,
    /// The same, for a nucleic acid.
    ///
    /// Separate because the two are drawn at different scales and always were:
    /// a nucleic backbone ribbon has to hold its own against base rings several
    /// ångströms across, where a protein ribbon only has to read against itself.
    /// Sharing one ratio left the duplex looking spindly beside Mol\*'s.
    pub nucleic_aspect_ratio: f32,
    /// How much wider than the strand an arrowhead's back face is.
    pub arrow_factor: f32,
    /// Spline samples per residue.
    pub linear_segments: usize,
    /// Sides of a rounded cross-section. Ignored by the rectangular ones, which
    /// have four whatever this says.
    pub radial_segments: usize,
    /// Draw helices as round tubes rather than flat ribbons.
    pub tubular_helices: bool,
    /// Draw each nucleic base as the flat outline of its ring, on a stick —
    /// the ladder rungs.
    ///
    /// Without them a duplex is two bare ribbons and reads as nothing in
    /// particular; the rungs are what make it legible as base pairs. See
    /// `sweep_base`.
    pub base_rings: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            size_factor: 0.2,
            aspect_ratio: 5.0,
            nucleic_aspect_ratio: 8.0,
            arrow_factor: 1.5,
            linear_segments: 8,
            radial_segments: 16,
            tubular_helices: false,
            base_rings: true,
        }
    }
}

/// The arrays a cartoon reads, already decoded.
///
/// Assembled by each backend's actor kind from what the actor binds, so the
/// curve itself never touches a `DataStore` or a `Subset`. `sse` and
/// `chain_of_residue` may be empty, which is not an error: a structure with no
/// secondary-structure assignment is all coil, and one with no chain column is
/// one chain.
pub struct Backbone<'a> {
    /// Atom centres.
    pub positions: &'a [Vec3],
    /// Which residue each atom belongs to.
    pub residue_of_atom: &'a [u32],
    /// Which distinct name each atom has, as an index into `names`.
    pub name_of_atom: &'a [u32],
    /// The distinct atom names, as uploaded.
    pub names: &'a [String],
    /// One secondary-structure code per residue. See `Form::of_code`.
    pub sse: &'a [u8],
    /// Which chain each residue sits in.
    pub chain_of_residue: &'a [u32],
}

/// A swept cartoon, as vertex arrays.
///
/// Not a `Mesh`: what attributes to put on the GPU is the backend's choice, and
/// one of the two never writes colours at all. `residue` is what makes a colour
/// change a repaint rather than a rebuild — it says which residue each vertex
/// came from, so a per-residue colour expands to per-vertex without
/// re-tessellating.
#[derive(Debug, Default)]
pub struct Ribbon {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    /// Which residue each vertex came from, parallel to `positions`.
    pub residue: Vec<u32>,
}

impl Ribbon {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// What a stretch of backbone is drawn as.
///
/// Fewer states than the wire carries, because this is about geometry: a 3-10
/// helix and a pi helix are drawn as helices, and nothing distinguishes a turn
/// from a bend once both are tubes. The eight-state code is still what travels,
/// so a later renderer that *does* want them apart loses nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// A round tube. Also what an unassigned residue gets.
    Coil,
    /// A flat ribbon, or a round tube under [`Style::tubular_helices`].
    Helix,
    /// A flat rectangular ribbon.
    Strand,
    /// The tapering head on the last residue of a strand run.
    Arrow,
    /// A flat ribbon with the frame swapped, so it sits edge-on to the bases.
    Nucleic,
}

impl Form {
    /// Reads one secondary-structure code as it arrives on the wire.
    ///
    /// The codes are DSSP's eight states, but what fills them today is
    /// biotite's P-SEA, which only ever reports helix, strand and nothing. So
    /// the unassigned state has to be ordinary rather than exceptional: it is
    /// coil, which is what an unassigned residue looks like.
    fn of_code(code: u8) -> Self {
        match code {
            1 | 4 | 5 => Form::Helix,
            2 | 3 => Form::Strand,
            _ => Form::Coil,
        }
    }

    /// Half-width across the ribbon and half-thickness through it.
    fn size(self, style: &Style) -> (f32, f32) {
        let thick = style.size_factor.max(0.001);
        let wide = thick * style.aspect_ratio.max(1.0);
        match self {
            Form::Nucleic => (thick * style.nucleic_aspect_ratio.max(1.0), thick),
            Form::Coil => (thick, thick),
            // A tube wide enough to read as a helix rather than as a fat coil.
            // The 1.5 is Mol*'s.
            Form::Helix if style.tubular_helices => {
                let radius = wide * 1.5;
                (radius, radius)
            }
            Form::Helix | Form::Strand => (wide, thick),
            // Never asked: an arrow's width is driven along its length rather
            // than taken from one target. See `pair_sizes`.
            Form::Arrow => (wide * style.arrow_factor.max(1.0), thick),
        }
    }

    /// Which cross-section this form is swept with.
    fn profile(self, style: &Style) -> Profile {
        match self {
            Form::Coil => Profile::rounded(style.radial_segments),
            Form::Helix if style.tubular_helices => Profile::rounded(style.radial_segments),
            // Elliptical rather than square: it is Mol*'s default helix profile
            // and PyMOL's, and a hard-edged helix reads as a folded strip.
            Form::Helix => Profile::rounded(style.radial_segments),
            Form::Strand | Form::Arrow | Form::Nucleic => Profile::rectangular(),
        }
    }
}

/// Which polymer a residue belongs to.
///
/// Decided by which trace atom was found, not by any annotation. A residue with
/// a `CA` is a protein residue whatever a file calls it, and that is the only
/// test that stays right for a modified residue or a non-standard name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Polymer {
    Protein,
    Nucleic,
}

impl Polymer {
    /// How far apart two consecutive trace atoms may be before the chain is
    /// treated as broken.
    ///
    /// Consecutive CAs sit 3.8 Å apart, and consecutive phosphates about 6.5 Å.
    /// Both limits are generous: the cost of splining across a genuine gap is a
    /// ribbon running through empty space, which is worse than a visible break.
    fn gap(self) -> f32 {
        match self {
            Polymer::Protein => 5.0,
            Polymer::Nucleic => 9.0,
        }
    }
}

/// Which of a residue's two guide atoms a name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Guide {
    /// The point the curve passes through.
    Trace,
    /// The atom the trace points at, fixing which way the ribbon faces.
    Direction,
}

/// What role an atom name plays, and how much it is preferred.
///
/// The rank breaks ties between fallbacks: `P` is the nucleic trace when it is
/// there, and `C3'` only when it is not — the first residue of a chain has no
/// phosphate.
///
/// `CA` is also the name of a calcium ion, which would make a one-residue
/// segment out of every bound calcium. Nothing guards against it here because
/// nothing needs to: a segment of one residue has no interval to sample and is
/// dropped.
fn guide_role(name: &str) -> Option<(Polymer, Guide, u8)> {
    // Primes are written `'` in mmCIF and `*` in older PDB files.
    match name {
        "CA" => Some((Polymer::Protein, Guide::Trace, 0)),
        "O" => Some((Polymer::Protein, Guide::Direction, 0)),
        // The C-terminal residue has a carboxyl rather than a carbonyl.
        "OXT" | "OT1" => Some((Polymer::Protein, Guide::Direction, 1)),
        "P" => Some((Polymer::Nucleic, Guide::Trace, 0)),
        "C3'" | "C3*" => Some((Polymer::Nucleic, Guide::Trace, 1)),
        "C4'" | "C4*" => Some((Polymer::Nucleic, Guide::Trace, 2)),
        // Into the base, which is PyMOL's choice of nucleic reference atom.
        "C2" => Some((Polymer::Nucleic, Guide::Direction, 0)),
        "O5'" | "O5*" => Some((Polymer::Nucleic, Guide::Direction, 1)),
        "C1'" | "C1*" => Some((Polymer::Nucleic, Guide::Direction, 2)),
        _ => None,
    }
}

/// Which of the base-ring atoms a name is, in the order [`Base`] reads them:
/// N1, C2, N3, C4, C5, C6, N7, C8, N9.
///
/// `C2` is also the nucleic direction atom. Both lookups run over every atom, so
/// it simply fills two roles rather than needing a rule about which wins.
fn base_slot(name: &str) -> Option<usize> {
    Some(match name {
        "N1" => 0,
        "C2" => 1,
        "N3" => 2,
        "C4" => 3,
        "C5" => 4,
        "C6" => 5,
        "N7" => 6,
        "C8" => 7,
        "N9" => 8,
        _ => return None,
    })
}

/// The most atoms a base's outline can have: a purine's fused bicyclic.
const RING_ATOMS: usize = 9;

/// One nucleic base, as the outline of its ring system.
///
/// This is Mol\*'s `nucleotide-ring` rather than its `nucleotide-block`. The
/// block draws every base as the same standard-sized rectangle; the ring follows
/// the actual atoms, so a purine is visibly the larger fused shape and a
/// pyrimidine a plain hexagon. It is what Mol\*'s default preset uses, and it is
/// the more honest picture — the outline is measured rather than stipulated.
#[derive(Debug, Clone, Copy)]
struct Base {
    /// The perimeter of the ring system, in order around the outside. Only the
    /// first [`Base::corners`] entries are used.
    perimeter: [Vec3; RING_ATOMS],
    corners: usize,
    /// The glycosidic nitrogen, where the stick to the backbone starts.
    attach: Vec3,
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
    fn read(found: &[Option<Vec3>; RING_ATOMS]) -> Option<Self> {
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
    fn outline(&self) -> &[Vec3] {
        &self.perimeter[..self.corners]
    }

    /// The plane the ring lies in, as a centre and a unit normal.
    ///
    /// Newell's method over the whole outline rather than a cross product of
    /// three atoms: a ring is only approximately planar, and picking three of
    /// its nine atoms would let a single displaced one tilt the whole face.
    /// Newell's averages every edge and is exact for a planar polygon.
    fn plane(&self) -> Option<(Vec3, Vec3)> {
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
struct Node {
    residue: u32,
    position: Vec3,
    /// Towards the direction atom, unnormalised. Zero when the residue had no
    /// direction atom at all, which [`orient`] fills in.
    direction: Vec3,
    form: Form,
    polymer: Polymer,
    chain: u32,
    /// The base to hang off this residue, for a nucleic one whose ring is
    /// complete. Always `None` for a protein residue.
    base: Option<Base>,
}

/// A run of residues the curve is continuous through.
type Segment = Vec<Node>;

/// The decoded arrays, kept alive so [`Backbone`] can borrow them.
///
/// Reading an actor's bindings into this lives here rather than in a backend
/// because it is not a pipeline decision: both pathways want exactly the same
/// six arrays, read exactly the same way. What they do with the resulting
/// [`Ribbon`] is where they part company.
pub struct Input {
    positions: Vec<Vec3>,
    residue_of_atom: Vec<u32>,
    name_of_atom: Vec<u32>,
    names: Vec<String>,
    sse: Vec<u8>,
    chain_of_residue: Vec<u32>,
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
fn read(request: &Request) -> Option<Input> {
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

/// Builds the cartoon for one structure.
pub fn build(backbone: &Backbone, style: &Style) -> Ribbon {
    let mut ribbon = Ribbon::default();
    let (mut residues, mut nucleic, mut bases) = (0usize, 0usize, 0usize);
    for mut segment in segments(backbone) {
        orient(&mut segment);
        smooth_strands(&mut segment);
        let samples = sample(&segment, style);
        sweep(&samples, style, &mut ribbon);

        residues += segment.len();
        nucleic += segment
            .iter()
            .filter(|node| node.polymer == Polymer::Nucleic)
            .count();
        for node in &segment {
            // The trace point rather than a sample: a cardinal spline passes
            // through its control points, so the guide position is on the
            // ribbon and the stick meets it rather than floating beside it.
            if let Some(base) = node.base {
                bases += 1;
                if style.base_rings {
                    sweep_base(&base, node.position, node.residue, style, &mut ribbon);
                }
            }
        }
    }
    // A nucleic residue whose ring did not resolve draws a bare ribbon, which
    // looks like a rendering bug rather than like missing atoms. Counting both
    // is what tells the two apart without opening the file.
    debug!("draw: cartoon over {residues} residues ({nucleic} nucleic, {bases} with a base ring)");
    ribbon
}

/// Draws one nucleic base as a flat ring on a stick — one rung of the ladder.
///
/// Two closed solids, both swept by [`sweep_run`] over two samples each, which
/// is the whole reason no prism or cylinder primitive is needed: a ring *is* its
/// own outline swept a short way along its normal and capped, and a stick is a
/// round profile swept along its axis. Both come out closed for free, which the
/// moment pathway requires.
///
/// The outline is measured from the atoms, so the shape carries information: a
/// purine really is bigger than a pyrimidine on screen, and a distorted ring
/// looks distorted.
fn sweep_base(base: &Base, trace: Vec3, residue: u32, style: &Style, ribbon: &mut Ribbon) {
    let thickness = style.size_factor.max(0.001);

    if let Some((centre, normal)) = base.plane() {
        // The ring is swept *along its own normal*, so the profile's two axes
        // both lie in the base plane and its coordinates are plain ångströms —
        // which is why the half-extents below are 1.
        let across = normal.any_orthonormal_vector();
        let up = normal.cross(across).normalize_or_zero();

        let mut outline: Vec<Vec2> = base
            .outline()
            .iter()
            .map(|corner| {
                let offset = *corner - centre;
                Vec2::new(offset.dot(across), offset.dot(up))
            })
            .collect();
        // The profile wants a counter-clockwise outline, and which way round the
        // atoms run depends on which face of the base happens to be up. Twice
        // the signed area says, and costs one pass.
        let twice_area: f32 = (0..outline.len())
            .map(|corner| {
                let (from, to) = (outline[corner], outline[(corner + 1) % outline.len()]);
                from.x * to.y - to.x * from.y
            })
            .sum();
        if twice_area < 0.0 {
            outline.reverse();
        }

        let face = |position: Vec3| Sample {
            position,
            tangent: normal,
            across,
            up,
            residue,
            half_width: 1.0,
            half_thick: 1.0,
            // Never read by the sweep, which takes the profile from its caller.
            form: Form::Nucleic,
        };
        sweep_run(
            &[
                face(centre - normal * thickness),
                face(centre + normal * thickness),
            ],
            &Profile::polygon(outline),
            ribbon,
        );
    }

    // The stick, from the glycosidic nitrogen back to the backbone.
    let stick = trace - base.attach;
    let length = stick.length();
    if length < f32::EPSILON {
        return;
    }
    let axis = stick / length;
    let side = axis.any_orthonormal_vector();
    let radius = thickness;
    let joint = |position: Vec3| Sample {
        position,
        tangent: axis,
        across: side,
        up: axis.cross(side).normalize_or(Vec3::Y),
        residue,
        half_width: radius,
        half_thick: radius,
        form: Form::Nucleic,
    };
    sweep_run(
        &[joint(base.attach), joint(trace)],
        &Profile::rounded(style.radial_segments),
        ribbon,
    );
}

/// Groups the atoms into residues, then the residues into runs the curve is
/// continuous through.
fn segments(backbone: &Backbone) -> Vec<Segment> {
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
fn nodes(backbone: &Backbone) -> Vec<Node> {
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
fn orient(segment: &mut Segment) {
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
fn smooth_strands(segment: &mut Segment) {
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

/// One point along the curve, with the frame and the size the sweep needs.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    position: Vec3,
    tangent: Vec3,
    /// Across the ribbon: the wide axis.
    across: Vec3,
    /// Through the ribbon: the thin axis, and the flat face's normal.
    up: Vec3,
    residue: u32,
    half_width: f32,
    half_thick: f32,
    form: Form,
}

impl Sample {
    /// One frame of a sweep that is not a backbone run.
    ///
    /// For a sibling module building a closed solid out of the same machinery —
    /// a base ring, a glycan shape. `Sample::form` is read only when splitting
    /// a run at a change of secondary structure, which such a sweep never does,
    /// so it takes a fixed value that nothing looks at.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        position: Vec3,
        tangent: Vec3,
        across: Vec3,
        up: Vec3,
        residue: u32,
        half_width: f32,
        half_thick: f32,
    ) -> Self {
        Self {
            position,
            tangent,
            across,
            up,
            residue,
            half_width,
            half_thick,
            form: Form::Coil,
        }
    }
}

/// Tension inside a secondary-structure element.
///
/// Higher holds the curve closer to the guide points, which is what keeps a
/// helix looking like a helix instead of a smoothed sausage.
const ELEMENT_TENSION: f32 = 0.9;
/// Tension at the ends of one, where a looser curve joins the next without a
/// corner. Mol\* pins the boundary to this value; 0.5 is plain Catmull-Rom.
const JOIN_TENSION: f32 = 0.5;

/// Samples the spline through a segment, frame and size included.
fn sample(segment: &Segment, style: &Style) -> Vec<Sample> {
    let count = segment.len();
    let steps = style.linear_segments.max(1);
    let forms = pair_forms(segment);
    let tensions = tensions(segment);

    let mut samples = Vec::with_capacity((count - 1) * steps + 1);
    for pair in 0..count - 1 {
        // Four control points, with the ends of the segment repeated. Repeating
        // rather than extrapolating keeps the curve inside the structure.
        let p0 = segment[pair.saturating_sub(1)].position;
        let p1 = segment[pair].position;
        let p2 = segment[pair + 1].position;
        let p3 = segment[(pair + 2).min(count - 1)].position;

        let (start, end) = (segment[pair], segment[pair + 1]);
        let form = forms[pair];
        for step in 0..steps {
            let t = step as f32 / steps as f32;
            let tension = tensions[pair].lerp(tensions[pair + 1], t);
            samples.push(frame(
                cardinal(p0, p1, p2, p3, t, tension),
                tangent(p0, p1, p2, p3, t, tension),
                start.direction.lerp(end.direction, t),
                // The residue a sample is nearer, so a per-residue colour lands
                // where the eye expects the boundary.
                if t < 0.5 { start.residue } else { end.residue },
                form,
                start.polymer,
                pair_size(segment, &forms, pair, t, style),
            ));
        }
    }

    // The final endpoint, which the loop above stops one short of.
    let last = count - 1;
    let p0 = segment[last.saturating_sub(2).min(last)].position;
    let p1 = segment[last.saturating_sub(1)].position;
    let p2 = segment[last].position;
    samples.push(frame(
        p2,
        (p2 - p1).normalize_or(Vec3::Z),
        segment[last].direction,
        segment[last].residue,
        forms[last - 1],
        segment[last].polymer,
        pair_size(segment, &forms, last - 1, 1.0, style),
    ));
    let _ = p0;

    average_frames(&mut samples);
    steady_arrows(&mut samples);
    samples
}

/// Stops an arrowhead rolling along its own length.
///
/// Everywhere else a rotating frame is the data talking: the ribbon follows the
/// peptide planes, and a sheet that twists really is twisted. An arrowhead is
/// different because it is not a stretch of backbone at all — it is a *symbol*
/// for the end of one, a flat plate a residue long. Letting it roll makes the
/// head appear to wring itself round the tip, which is the most visible artefact
/// on a beta sheet because the head is the widest part of the ribbon.
///
/// The roll is frozen to the frame the arrow starts with, which is also the one
/// the strand behind it ends with, so the join stays exact. The axes are
/// re-orthogonalised against each sample's own tangent rather than copied
/// outright: the curve still bends through the head, and only the rotation
/// about it is held.
fn steady_arrows(samples: &mut [Sample]) {
    let mut start = 0;
    while start < samples.len() {
        if samples[start].form != Form::Arrow {
            start += 1;
            continue;
        }
        let mut end = start;
        while end + 1 < samples.len() && samples[end + 1].form == Form::Arrow {
            end += 1;
        }

        let held = samples[start].across;
        for sample in &mut samples[start..=end] {
            let tangent = sample.tangent;
            let across = (held - tangent * held.dot(tangent)).normalize_or_zero();
            if across.length_squared() < f32::EPSILON {
                // The head turned through the frame's own axis. Nothing to hold
                // on to, so leave the interpolated frame alone.
                continue;
            }
            sample.across = across;
            sample.up = tangent.cross(across).normalize_or(sample.up);
        }
        start = end + 1;
    }
}

/// What each interval between two residues is drawn as.
///
/// An interval takes the form of the residue it starts at, so a change of form
/// lands exactly on a guide point. The one exception is the arrowhead: the last
/// interval of a strand run becomes [`Form::Arrow`], which is what puts the
/// taper on the final residue rather than past the end of the sheet.
fn pair_forms(segment: &Segment) -> Vec<Form> {
    let mut forms: Vec<Form> = segment[..segment.len() - 1]
        .iter()
        .map(|node| node.form)
        .collect();
    for pair in 0..forms.len() {
        let ends_a_strand = segment[pair].form == Form::Strand
            && segment[pair + 1].form == Form::Strand
            && segment
                .get(pair + 2)
                .is_none_or(|next| next.form != Form::Strand);
        if ends_a_strand {
            forms[pair] = Form::Arrow;
        }
    }
    forms
}

/// Spline tension per residue: loose where an element starts or ends, tight
/// inside one.
fn tensions(segment: &Segment) -> Vec<f32> {
    (0..segment.len())
        .map(|index| {
            let form = segment[index].form;
            let boundary = index == 0
                || index == segment.len() - 1
                || segment[index - 1].form != form
                || segment[index + 1].form != form;
            if boundary {
                JOIN_TENSION
            } else {
                ELEMENT_TENSION
            }
        })
        .collect()
}

/// Half-width and half-thickness partway along one interval.
///
/// Interpolated between the two residues' own targets, so a helix meeting a coil
/// tapers over the residue between them and the two runs meet at exactly the
/// same size.
///
/// An arrowhead is the exception: its width is driven from the wide back face
/// down to nothing. That is a step against the strand before it, and it is why
/// the arrow is swept as its own capped run. The step is not infinitely sharp —
/// [`sweep`] shares the boundary sample between runs, so the strand flares out
/// to the arrow's width over its own last segment, about an eighth of a residue.
/// The result is a bevelled shoulder rather than a flat annulus, which is both
/// what the sharing buys and cheaper than the alternative.
fn pair_size(segment: &Segment, forms: &[Form], pair: usize, t: f32, style: &Style) -> (f32, f32) {
    let (_, thick) = segment[pair].form.size(style);
    if forms[pair] == Form::Arrow {
        let (wide, _) = Form::Arrow.size(style);
        // Not quite to zero: a degenerate tip would give the cap no area and
        // leave the normals there undefined.
        return (wide * (1.0 - t) + thick * 0.1 * t, thick);
    }
    let (start_wide, start_thick) = segment[pair].form.size(style);
    let (end_wide, end_thick) = segment[pair + 1].form.size(style);
    (start_wide.lerp(end_wide, t), start_thick.lerp(end_thick, t))
}

/// Builds one sample's frame from the curve and the residue direction.
#[allow(clippy::too_many_arguments)]
fn frame(
    position: Vec3,
    tangent: Vec3,
    direction: Vec3,
    residue: u32,
    form: Form,
    polymer: Polymer,
    (half_width, half_thick): (f32, f32),
) -> Sample {
    // The wide axis lies in the peptide plane, perpendicular to the curve. For a
    // beta strand that is very nearly the carbonyl direction, which is why the
    // flat face comes out parallel to the sheet.
    let mut across = (direction - tangent * direction.dot(tangent)).normalize_or_zero();
    if across.length_squared() < f32::EPSILON {
        // The direction was parallel to the curve, or there was none. Any
        // perpendicular will do; this one is stable under small changes of
        // tangent, which an arbitrary axis is not.
        across = tangent.any_orthonormal_vector();
    }
    let mut up = tangent.cross(across).normalize_or(Vec3::Y);

    // Mol* swaps the frame for nucleic acids. With the direction pointing into
    // the base, swapping is what leaves the ribbon edge-on to the bases rather
    // than lying flat against them.
    if polymer == Polymer::Nucleic {
        std::mem::swap(&mut across, &mut up);
    }

    Sample {
        position,
        tangent,
        across,
        up,
        residue,
        half_width,
        half_thick,
        form,
    }
}

/// The three-point averaging pass Mol\* applies to the frames.
///
/// Each interior frame's wide axis becomes the mean of itself and its
/// neighbours', which removes the small discontinuities left by interpolating
/// the residue directions independently. The axes are made orthogonal to the
/// tangent again afterwards, or the cross-section would shear.
fn average_frames(samples: &mut [Sample]) {
    if samples.len() < 3 {
        return;
    }
    let original: Vec<Vec3> = samples.iter().map(|sample| sample.across).collect();
    for index in 1..samples.len() - 1 {
        let mean = original[index - 1] + original[index] + original[index + 1];
        let sample = &mut samples[index];
        let tangent = sample.tangent;
        let across = (mean - tangent * mean.dot(tangent)).normalize_or_zero();
        if across.length_squared() < f32::EPSILON {
            continue;
        }
        sample.across = across;
        sample.up = tangent.cross(across).normalize_or(sample.up);
    }
}

/// A cardinal spline through four control points.
///
/// `tension` of 0.5 is Catmull-Rom. This is Mol\*'s `v3spline`, which is the
/// standard form.
fn cardinal(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    let (t2, t3) = (t * t, t * t * t);
    let v0 = (p2 - p0) * tension;
    let v1 = (p3 - p1) * tension;
    (p1 * 2.0 - p2 * 2.0 + v0 + v1) * t3 + (p1 * -3.0 + p2 * 3.0 - v0 * 2.0 - v1) * t2 + v0 * t + p1
}

/// The curve direction, by central difference.
///
/// Differencing rather than differentiating: the analytic derivative vanishes
/// wherever three control points are collinear and the tension cancels, and a
/// zero tangent takes the whole frame with it. A finite difference over a real
/// interval cannot.
fn tangent(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    const DELTA: f32 = 0.01;
    let before = cardinal(p0, p1, p2, p3, t - DELTA, tension);
    let after = cardinal(p0, p1, p2, p3, t + DELTA, tension);
    (after - before).normalize_or(Vec3::Z)
}

/// One point of a closed cross-section, in units of half-width and
/// half-thickness.
#[derive(Debug, Clone, Copy)]
pub(super) struct Rim {
    /// Across the ribbon, in -1..1.
    u: f32,
    /// Through the ribbon, in -1..1.
    v: f32,
    /// The outward normal in the same two axes, or `None` to take the ellipse
    /// gradient at this point — which is `(u / half_width, v / half_thick)`, and
    /// so cannot be precomputed because it depends on the size at each sample.
    normal: Option<Vec2>,
}

/// A closed cross-section.
///
/// Two lists rather than one because a hard edge needs a point twice, once per
/// face, and a cap needs it once. `rim` is what the sides are built from and
/// carries the duplicates; `outline` is what a cap is fanned over and does not.
/// Both are wound the same way — counter-clockwise in the across-up plane, which
/// with the sweep advancing along the tangent puts the front faces outward.
pub struct Profile {
    rim: Vec<Rim>,
    outline: Vec<Vec2>,
}

impl Profile {
    /// An ellipse, for a coil or a helix.
    pub fn rounded(sides: usize) -> Self {
        let sides = sides.clamp(3, 64);
        let point = |index: usize| {
            let angle = std::f32::consts::TAU * index as f32 / sides as f32;
            Vec2::new(angle.cos(), angle.sin())
        };
        Self {
            rim: (0..sides)
                .map(|index| {
                    let at = point(index);
                    Rim {
                        u: at.x,
                        v: at.y,
                        normal: None,
                    }
                })
                .collect(),
            outline: (0..sides).map(point).collect(),
        }
    }

    /// A rectangle, for a strand or a nucleic ribbon.
    pub fn rectangular() -> Self {
        Self::polygon(vec![
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(-1.0, -1.0),
        ])
    }

    /// Any flat-sided closed outline, wound counter-clockwise.
    ///
    /// Each face gets its own pair of rim points so the edges stay hard. That
    /// leaves a zero-area quad at each corner, which costs a few degenerate
    /// triangles and no fragments, and is worth it for not needing a second
    /// sweep path.
    ///
    /// The face normal is **derived from the edge** rather than tabulated. The
    /// hand-written table this replaces was off by one, which gave every
    /// rectangular cross-section its neighbour's normal: a beta strand's broad
    /// face was shaded as though it were the thin edge, and the ribbon picked up
    /// flat washed-out bands wherever it turned over.
    pub(super) fn polygon(outline: Vec<Vec2>) -> Self {
        let sides = outline.len();
        let mut rim = Vec::with_capacity(sides * 2);
        for side in 0..sides {
            let (from, to) = (outline[side], outline[(side + 1) % sides]);
            let along = (to - from).normalize_or_zero();
            // The outward normal of a counter-clockwise outline is its edge
            // turned a quarter turn clockwise.
            let normal = Vec2::new(along.y, -along.x);
            rim.push(Rim {
                u: from.x,
                v: from.y,
                normal: Some(normal),
            });
            rim.push(Rim {
                u: to.x,
                v: to.y,
                normal: Some(normal),
            });
        }
        Self { rim, outline }
    }
}

/// Sweeps each run of samples through its own profile, closed and capped.
///
/// Runs are split where the form changes, which is also where the width steps if
/// it steps at all. Each is closed on its own, and neighbours abut at a shared
/// sample so no gap shows.
fn sweep(samples: &[Sample], style: &Style, ribbon: &mut Ribbon) {
    let mut start = 0;
    while start < samples.len() {
        let form = samples[start].form;
        let mut end = start;
        while end + 1 < samples.len() && samples[end + 1].form == form {
            end += 1;
        }
        // Inclusive of the next run's first sample, so the two meet exactly.
        let stop = (end + 2).min(samples.len());
        if stop - start >= 2 {
            if form == Form::Arrow && stop == end + 2 {
                sweep_arrow(&samples[start..stop], &form.profile(style), ribbon);
            } else {
                sweep_run(&samples[start..stop], &form.profile(style), ribbon);
            }
        }
        start = end + 1;
    }
}

/// Sweeps an arrowhead, whose shared boundary sample has to be treated
/// differently from every other run's.
///
/// Sharing the next run's first sample is what makes two runs meet without a
/// gap, and it is right everywhere the width is continuous across the join. At
/// an arrowhead it is not: the taper reaches its narrowest at the last sample of
/// the arrow's own interval, and the shared sample carries the *following*
/// run's width — a coil's full radius. Swept as one run that reads as a point
/// that immediately flares back out, which is a bow-tie rather than an arrow,
/// and any roll across those two samples shows up as a twist in the pinch.
///
/// So the boundary sample is used for its **position** and given the arrow's own
/// tip width instead. The next run still starts from the unmodified sample, so
/// nothing gains a gap: the two simply stop agreeing about width, which is the
/// truth at the end of a sheet.
///
/// Its frame is taken from the sample before it as well. An arrowhead is a flat
/// plate — a quarter turn of roll across the last half ångström of one is never
/// what the data means.
fn sweep_arrow(run: &[Sample], profile: &Profile, ribbon: &mut Ribbon) {
    let mut tapered: Vec<Sample> = run.to_vec();
    let last = tapered.len() - 1;
    let before = tapered[last - 1];
    let tip = &mut tapered[last];
    // Not to zero: a degenerate cross-section leaves the cap with no area and
    // its normals undefined.
    tip.half_width = before.half_width * 0.15;
    tip.half_thick = before.half_thick;
    tip.across = before.across;
    tip.up = before.up;
    sweep_run(&tapered, profile, ribbon);
}

/// Sweeps one profile along one run, and caps both ends.
pub fn sweep_run(run: &[Sample], profile: &Profile, ribbon: &mut Ribbon) {
    let sides = profile.rim.len();
    let base = ribbon.positions.len() as u32;

    for sample in run {
        for rim in &profile.rim {
            let offset =
                sample.across * rim.u * sample.half_width + sample.up * rim.v * sample.half_thick;
            let flat = rim.normal.unwrap_or_else(|| {
                // The outward normal of the ellipse this rim point lies on.
                Vec2::new(
                    rim.u / sample.half_width.max(1e-6),
                    rim.v / sample.half_thick.max(1e-6),
                )
            });
            let normal = (sample.across * flat.x + sample.up * flat.y).normalize_or(sample.up);
            push(ribbon, sample.position + offset, normal, sample.residue);
        }
    }

    // Sides. Wound so that the face normal comes out along `rim x tangent`,
    // which is outward — see the note on `Profile`.
    for step in 0..run.len() - 1 {
        for side in 0..sides {
            let next = (side + 1) % sides;
            let a = base + (step * sides + side) as u32;
            let b = base + (step * sides + next) as u32;
            let c = base + ((step + 1) * sides + next) as u32;
            let d = base + ((step + 1) * sides + side) as u32;
            ribbon.indices.extend([a, b, c, a, c, d]);
        }
    }

    // Caps, or the moment pathway reads the run as a solid with no far wall and
    // draws it far too clear. Each is a fan from the centre with one flat
    // normal, so the ends read as ends rather than as a continuation.
    cap(ribbon, &run[0], profile, true);
    cap(ribbon, &run[run.len() - 1], profile, false);
}

/// Closes one end of a run with a triangle fan.
///
/// `front` is the end the tangent points away from, whose normal is therefore
/// the reverse of it. The outline is wound counter-clockwise about the tangent,
/// so the front cap takes it reversed and the back cap as it is.
fn cap(ribbon: &mut Ribbon, sample: &Sample, profile: &Profile, front: bool) {
    let normal = if front {
        -sample.tangent
    } else {
        sample.tangent
    };
    let centre = ribbon.positions.len() as u32;
    push(ribbon, sample.position, normal, sample.residue);

    for point in &profile.outline {
        let offset =
            sample.across * point.x * sample.half_width + sample.up * point.y * sample.half_thick;
        push(ribbon, sample.position + offset, normal, sample.residue);
    }

    let sides = profile.outline.len() as u32;
    for side in 0..sides {
        let a = centre + 1 + side;
        let b = centre + 1 + (side + 1) % sides;
        if front {
            ribbon.indices.extend([centre, b, a]);
        } else {
            ribbon.indices.extend([centre, a, b]);
        }
    }
}

fn push(ribbon: &mut Ribbon, position: Vec3, normal: Vec3, residue: u32) {
    ribbon.positions.push([position.x, position.y, position.z]);
    ribbon.normals.push([normal.x, normal.y, normal.z]);
    ribbon.residue.push(residue);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An idealised alpha helix: 3.6 residues per turn, 1.5 Å rise, 2.3 Å
    /// radius. Real enough to exercise the frames without reading a file.
    fn helix(residues: usize) -> (Vec<Vec3>, Vec<u32>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..residues {
            let angle = std::f32::consts::TAU * index as f32 / 3.6;
            let ca = Vec3::new(2.3 * angle.cos(), 1.5 * index as f32, 2.3 * angle.sin());
            // The carbonyl points along the helix axis, roughly, and alternates
            // in the raw coordinates the way a real one does.
            let flip = if index % 2 == 0 { 1.0 } else { -1.0 };
            positions.push(ca);
            residue_of_atom.push(index as u32);
            name_of_atom.push(0);
            positions.push(ca + Vec3::new(angle.cos(), 0.6 * flip, angle.sin()) * flip);
            residue_of_atom.push(index as u32);
            name_of_atom.push(1);
        }
        (positions, residue_of_atom, name_of_atom)
    }

    const NAMES: [&str; 2] = ["CA", "O"];

    fn names() -> Vec<String> {
        NAMES.iter().map(|name| name.to_string()).collect()
    }

    fn build_helix(residues: usize, sse: &[u8], style: &Style) -> Ribbon {
        let (positions, residue_of_atom, name_of_atom) = helix(residues);
        let names = names();
        build(
            &Backbone {
                positions: &positions,
                residue_of_atom: &residue_of_atom,
                name_of_atom: &name_of_atom,
                names: &names,
                sse,
                chain_of_residue: &[],
            },
            style,
        )
    }

    #[test]
    fn builds_a_helix() {
        let ribbon = build_helix(12, &[1; 12], &Style::default());
        assert!(!ribbon.is_empty());
        assert_eq!(ribbon.positions.len(), ribbon.normals.len());
        assert_eq!(ribbon.positions.len(), ribbon.residue.len());
        assert!(
            ribbon.positions.iter().flatten().all(|v| v.is_finite()),
            "every vertex should be finite"
        );
        assert!(
            ribbon.normals.iter().flatten().all(|v| v.is_finite()),
            "every normal should be finite"
        );
    }

    /// Every index must address a vertex that exists. A run that miscounts its
    /// base offset produces a mesh that renders as garbage or crashes the
    /// driver, and nothing else in the pipeline checks.
    #[test]
    fn indices_stay_in_range() {
        for sse in [vec![0u8; 10], vec![1; 10], vec![3; 10]] {
            let ribbon = build_helix(10, &sse, &Style::default());
            let vertices = ribbon.positions.len() as u32;
            assert!(
                ribbon.indices.iter().all(|index| *index < vertices),
                "an index escaped the vertex buffer"
            );
            assert_eq!(ribbon.indices.len() % 3, 0);
        }
    }

    /// The mesh has to be closed, or the moment pathway draws it as a solid
    /// whose far wall is infinitely distant — and that fails *quietly*, reading
    /// as too clear rather than as an error.
    ///
    /// Vertices are welded by position first, and that is not a convenience:
    /// closedness is a property of the surface, not of the vertex buffer. A cap
    /// meets the tube at a hard edge, so the two carry different normals and
    /// therefore have to be different vertices — an index-level test would call
    /// a perfectly closed ribbon open. Degenerate triangles are skipped too: the
    /// rectangular profile makes four of them per corner by design.
    #[test]
    fn every_run_is_closed() {
        for sse in [vec![0u8; 8], vec![1; 8], vec![3; 8]] {
            let ribbon = build_helix(8, &sse, &Style::default());

            // A tenth of a micron, which is far below anything the sweep
            // produces and far above the rounding between two ways of computing
            // the same point.
            let key = |index: u32| {
                let [x, y, z] = ribbon.positions[index as usize];
                let grid = |v: f32| (v * 10_000.0).round() as i64;
                (grid(x), grid(y), grid(z))
            };
            let mut welded: HashMap<(i64, i64, i64), u32> = HashMap::default();
            let mut canonical = Vec::with_capacity(ribbon.positions.len());
            for index in 0..ribbon.positions.len() as u32 {
                let next = welded.len() as u32;
                canonical.push(*welded.entry(key(index)).or_insert(next));
            }

            // Signed, so a face wound the wrong way shows up as a mismatch
            // rather than cancelling with its neighbour.
            let mut edges: HashMap<(u32, u32), i32> = HashMap::default();
            for triangle in ribbon.indices.chunks_exact(3) {
                let [a, b, c] = [
                    canonical[triangle[0] as usize],
                    canonical[triangle[1] as usize],
                    canonical[triangle[2] as usize],
                ];
                if a == b || b == c || a == c {
                    continue;
                }
                for (from, to) in [(a, b), (b, c), (c, a)] {
                    *edges.entry((from.min(to), from.max(to))).or_default() +=
                        if from < to { 1 } else { -1 };
                }
            }
            let open = edges.values().filter(|count| **count != 0).count();
            assert_eq!(open, 0, "{open} unpaired edges for sse {}", sse[0]);
        }
    }

    /// The flip correction is the difference between a ribbon and a corkscrew.
    ///
    /// A straight strand rather than the helix fixture, because a strand is
    /// where the alternation is real and total: consecutive carbonyls point
    /// almost exactly opposite ways, so the raw vectors reverse every residue
    /// and an uncorrected ribbon turns 180° per residue.
    #[test]
    fn flip_correction_keeps_the_frame_steady() {
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..10u32 {
            let ca = Vec3::new(index as f32 * 3.8, 0.0, 0.0);
            let flip = if index % 2 == 0 { 1.0 } else { -1.0 };
            positions.push(ca);
            residue_of_atom.push(index);
            name_of_atom.push(0);
            // The carbonyl, perpendicular to the strand and alternating.
            positions.push(ca + Vec3::new(0.0, 0.0, 1.2 * flip));
            residue_of_atom.push(index);
            name_of_atom.push(1);
        }
        let names = names();
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residue_of_atom,
            name_of_atom: &name_of_atom,
            names: &names,
            sse: &[3; 10],
            chain_of_residue: &[],
        };
        let mut segment = segments(&backbone).remove(0);

        // Confirm the input really does alternate before trusting the result —
        // a fixture that happened not to would pass the assertion below without
        // the correction doing anything.
        let raw: Vec<Vec3> = segment.iter().map(|node| node.direction).collect();
        assert!(
            raw.windows(2).all(|pair| pair[0].dot(pair[1]) < 0.0),
            "every consecutive pair in the fixture should be reversed"
        );

        orient(&mut segment);
        for pair in segment.windows(2) {
            assert!(
                pair[0].direction.dot(pair[1].direction) > 0.0,
                "consecutive directions should agree after correction"
            );
        }
    }

    /// A gap wider than a bond breaks the curve rather than being splined
    /// across, which would run a ribbon through empty space.
    #[test]
    fn a_gap_breaks_the_curve() {
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..8u32 {
            // Two runs of four, 40 Å apart.
            let shift = if index < 4 { 0.0 } else { 40.0 };
            positions.push(Vec3::new(index as f32 * 3.8 + shift, 0.0, 0.0));
            residue_of_atom.push(index);
            name_of_atom.push(0);
        }
        let names = names();
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residue_of_atom,
            name_of_atom: &name_of_atom,
            names: &names,
            sse: &[],
            chain_of_residue: &[],
        };
        assert_eq!(segments(&backbone).len(), 2);
    }

    /// A different chain is a different curve even when the trace atoms happen
    /// to be close, which they are at an interface.
    #[test]
    fn a_chain_change_breaks_the_curve() {
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..6u32 {
            positions.push(Vec3::new(index as f32 * 3.8, 0.0, 0.0));
            residue_of_atom.push(index);
            name_of_atom.push(0);
        }
        let names = names();
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residue_of_atom,
            name_of_atom: &name_of_atom,
            names: &names,
            sse: &[],
            chain_of_residue: &[0, 0, 0, 1, 1, 1],
        };
        assert_eq!(segments(&backbone).len(), 2);
    }

    /// A lone calcium ion is named `CA`, so it reaches the trace test. One
    /// residue has no interval to interpolate and must be dropped rather than
    /// drawn or panicked on.
    #[test]
    fn a_lone_calcium_draws_nothing() {
        let names = names();
        let ribbon = build(
            &Backbone {
                positions: &[Vec3::ZERO],
                residue_of_atom: &[0],
                name_of_atom: &[0],
                names: &names,
                sse: &[0],
                chain_of_residue: &[],
            },
            &Style::default(),
        );
        assert!(ribbon.is_empty());
    }

    /// A strand run ends in an arrowhead, which is wider than the sheet at its
    /// back face and tapers to nothing. The width is what says so.
    #[test]
    fn a_strand_ends_in_an_arrow() {
        let (positions, residue_of_atom, name_of_atom) = helix(8);
        let names = names();
        // Strand for the first six residues, coil after, so the run has an end
        // inside the segment.
        let sse = [3u8, 3, 3, 3, 3, 3, 0, 0];
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residue_of_atom,
            name_of_atom: &name_of_atom,
            names: &names,
            sse: &sse,
            chain_of_residue: &[],
        };
        let mut segment = segments(&backbone).remove(0);
        orient(&mut segment);
        let forms = pair_forms(&segment);
        assert_eq!(
            forms.iter().filter(|form| **form == Form::Arrow).count(),
            1,
            "exactly one interval should carry the arrowhead: {forms:?}"
        );

        let style = Style::default();
        let samples = sample(&segment, &style);
        let arrow: Vec<&Sample> = samples
            .iter()
            .filter(|sample| sample.form == Form::Arrow)
            .collect();
        assert!(!arrow.is_empty());
        let (strand_wide, _) = Form::Strand.size(&style);
        assert!(
            arrow[0].half_width > strand_wide,
            "the arrow's back face should be wider than the sheet"
        );
        assert!(
            arrow[arrow.len() - 1].half_width < arrow[0].half_width * 0.5,
            "the arrow should taper"
        );
    }

    /// An arrowhead narrows all the way to its tip and never widens again.
    ///
    /// The trap is absorbing the following run's first sample, which carries a
    /// *coil's* width: the head then tapers to a point and flares straight back
    /// out — a bow-tie, with the roll across the pinch reading as a twist.
    /// Walking the widths is what catches it: the shape is wrong long before
    /// any count or index is.
    #[test]
    fn an_arrowhead_never_widens() {
        // A nearly straight backbone with alternating carbonyls, which is what a
        // beta strand is — and an arrowhead only ever sits at the end of one.
        // The helix fixture is the wrong shape here: it turns about 100 degrees
        // per residue, so world-space frame rotation there is dominated by the
        // curve bending rather than by any roll, and the roll check below could
        // not tell the two apart.
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..10u32 {
            let ca = Vec3::new(index as f32 * 3.8, 0.0, 0.0);
            let flip = if index % 2 == 0 { 1.0 } else { -1.0 };
            positions.push(ca);
            residue_of_atom.push(index);
            name_of_atom.push(0);
            positions.push(ca + Vec3::new(0.0, 0.0, 1.2 * flip));
            residue_of_atom.push(index);
            name_of_atom.push(1);
        }
        let names = names();
        // A strand that ends inside the segment, so the arrow has a coil after
        // it — the case that flares if the following sample is absorbed.
        let sse = [3u8, 3, 3, 3, 3, 3, 0, 0, 0, 0];
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residue_of_atom,
            name_of_atom: &name_of_atom,
            names: &names,
            sse: &sse,
            chain_of_residue: &[],
        };
        let mut segment = segments(&backbone).remove(0);
        orient(&mut segment);
        let style = Style::default();
        let samples = sample(&segment, &style);

        let arrow: Vec<&Sample> = samples
            .iter()
            .filter(|sample| sample.form == Form::Arrow)
            .collect();
        assert!(arrow.len() >= 3, "expected a run of arrow samples");
        for pair in arrow.windows(2) {
            assert!(
                pair[1].half_width <= pair[0].half_width + 1e-6,
                "the head widened again: {} then {}",
                pair[0].half_width,
                pair[1].half_width
            );
        }

        // And the head does not roll. Beyond about a degree per sample the
        // twist is visible on a ribbon this wide.
        for pair in arrow.windows(2) {
            let turn = pair[0].across.dot(pair[1].across).clamp(-1.0, 1.0).acos();
            assert!(
                turn < 0.02,
                "the head rolled {:.1} degrees between samples",
                turn.to_degrees()
            );
        }
    }

    /// No secondary-structure column at all is the common case for anything
    /// P-SEA could not assign, and it must draw a plain tube rather than
    /// nothing.
    #[test]
    fn no_assignment_draws_a_coil() {
        let ribbon = build_helix(6, &[], &Style::default());
        assert!(!ribbon.is_empty());
    }

    /// Nucleic residues are found by their own trace atom and drawn as ribbons,
    /// with no secondary structure involved.
    #[test]
    fn builds_a_nucleic_ribbon() {
        let names: Vec<String> = ["P", "C2"].iter().map(|name| name.to_string()).collect();
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        let mut name_of_atom = Vec::new();
        for index in 0..8u32 {
            let angle = std::f32::consts::TAU * index as f32 / 10.0;
            let p = Vec3::new(9.0 * angle.cos(), index as f32 * 2.8, 9.0 * angle.sin());
            positions.push(p);
            residue_of_atom.push(index);
            name_of_atom.push(0);
            // Into the base, which is towards the helix axis.
            positions.push(p - Vec3::new(p.x, 0.0, p.z).normalize() * 4.0);
            residue_of_atom.push(index);
            name_of_atom.push(1);
        }
        let ribbon = build(
            &Backbone {
                positions: &positions,
                residue_of_atom: &residue_of_atom,
                name_of_atom: &name_of_atom,
                names: &names,
                sse: &[],
                chain_of_residue: &[],
            },
            &Style::default(),
        );
        assert!(!ribbon.is_empty());
        let vertices = ribbon.positions.len() as u32;
        assert!(ribbon.indices.iter().all(|index| *index < vertices));
    }

    /// A duplex with base rings, built from real-ish nucleotide geometry.
    ///
    /// Returns the arrays for `count` residues, each carrying a phosphate and a
    /// full purine ring system. `C2` does double duty as the ribbon's direction
    /// atom and as a ring atom, which is exactly how a real file has it.
    fn nucleotides(count: u32) -> (Vec<Vec3>, Vec<u32>, Vec<u32>, Vec<String>) {
        let names: Vec<String> = ["P", "N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"]
            .iter()
            .map(|name| name.to_string())
            .collect();
        let (mut positions, mut residues, mut atoms) = (Vec::new(), Vec::new(), Vec::new());
        for index in 0..count {
            let angle = std::f32::consts::TAU * index as f32 / 10.0;
            let p = Vec3::new(9.0 * angle.cos(), index as f32 * 3.4, 9.0 * angle.sin());
            // Inward, towards the helix axis, which is where the base sits.
            let inward = Vec3::new(-p.x, 0.0, -p.z).normalize();
            let side = inward.cross(Vec3::Y);
            // A flat, convex, roughly purine-shaped outline in the plane spanned
            // by `inward` and `side`, so the ring's normal comes out along the
            // helix axis as a real base's does. Order is by atom name, not
            // around the perimeter — `Base::read` is what knows the perimeter.
            let at = |out: f32, across: f32| p + inward * out + side * across;
            let ring = [
                at(6.5, 0.0),  // N1
                at(6.0, 1.0),  // C2
                at(5.0, 1.3),  // N3
                at(4.3, 0.7),  // C4
                at(4.6, -0.6), // C5
                at(5.9, -1.0), // C6
                at(3.9, -1.4), // N7
                at(2.9, -0.7), // C8
                at(3.2, 0.6),  // N9, the glycosidic nitrogen
            ];
            for (slot, position) in [p].iter().chain(ring.iter()).enumerate() {
                positions.push(*position);
                residues.push(index);
                atoms.push(slot as u32);
            }
        }
        (positions, residues, atoms, names)
    }

    /// Every nucleic residue gets a ring and a stick, and both are closed.
    ///
    /// Closure matters as much here as for the ribbon: a rung is drawn by the
    /// moment pathway as a solid, and an open one reads too clear rather than
    /// failing.
    #[test]
    fn nucleic_bases_become_closed_rungs() {
        let (positions, residues, atoms, names) = nucleotides(6);
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residues,
            name_of_atom: &atoms,
            names: &names,
            sse: &[],
            chain_of_residue: &[],
        };

        let bare = build(
            &backbone,
            &Style {
                base_rings: false,
                ..Style::default()
            },
        );
        let rungs = build(&backbone, &Style::default());
        assert!(
            rungs.positions.len() > bare.positions.len(),
            "switching the rings on should add geometry"
        );

        let vertices = rungs.positions.len() as u32;
        assert!(rungs.indices.iter().all(|index| *index < vertices));

        let key = |index: u32| {
            let [x, y, z] = rungs.positions[index as usize];
            let grid = |v: f32| (v * 10_000.0).round() as i64;
            (grid(x), grid(y), grid(z))
        };
        let mut welded: HashMap<(i64, i64, i64), u32> = HashMap::default();
        let mut canonical = Vec::with_capacity(rungs.positions.len());
        for index in 0..vertices {
            let next = welded.len() as u32;
            canonical.push(*welded.entry(key(index)).or_insert(next));
        }
        let mut edges: HashMap<(u32, u32), i32> = HashMap::default();
        for triangle in rungs.indices.chunks_exact(3) {
            let [a, b, c] = [
                canonical[triangle[0] as usize],
                canonical[triangle[1] as usize],
                canonical[triangle[2] as usize],
            ];
            if a == b || b == c || a == c {
                continue;
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                *edges.entry((from.min(to), from.max(to))).or_default() +=
                    if from < to { 1 } else { -1 };
            }
        }
        let open = edges.values().filter(|count| **count != 0).count();
        assert_eq!(open, 0, "{open} unpaired edges with base rings on");
    }

    /// A purine's outline is the nine-corner fused shape and a pyrimidine's the
    /// plain hexagon, and which one is decided by the ring rather than by the
    /// residue's name — so a modified base with the same ring system draws
    /// correctly and an unknown one is not a special case.
    ///
    /// The perimeter order is the point of the test. The atoms arrive numbered,
    /// and going round them in that order would cut a chord across the fused
    /// system instead of tracing its outside.
    #[test]
    fn purines_and_pyrimidines_differ_by_ring() {
        let named: [Vec3; RING_ATOMS] =
            std::array::from_fn(|slot| Vec3::new(slot as f32, 0.0, 0.0));
        let atoms = |five: bool| {
            let mut found: [Option<Vec3>; RING_ATOMS] =
                std::array::from_fn(|slot| Some(named[slot]));
            if !five {
                // N7, C8, N9 — the five-ring, which a pyrimidine lacks.
                found[6] = None;
                found[7] = None;
                found[8] = None;
            }
            found
        };

        let purine = Base::read(&atoms(true)).expect("a complete purine");
        assert_eq!(purine.corners, RING_ATOMS);
        assert_eq!(purine.attach, named[8], "a purine joins through N9");
        assert_eq!(
            purine.outline(),
            // N1 C2 N3 C4 N9 C8 N7 C5 C6 — round the six-ring to C4, across the
            // shared bond into the five-ring, and back out through C5.
            [
                named[0], named[1], named[2], named[3], named[8], named[7], named[6], named[4],
                named[5],
            ],
        );

        let pyrimidine = Base::read(&atoms(false)).expect("a complete pyrimidine");
        assert_eq!(pyrimidine.corners, 6);
        assert_eq!(pyrimidine.attach, named[0], "a pyrimidine joins through N1");
        assert_eq!(pyrimidine.outline(), &named[..6]);
    }

    /// An incomplete ring draws nothing rather than a misshapen outline. A
    /// partly resolved base is common in low-resolution structures.
    #[test]
    fn an_incomplete_ring_draws_no_base() {
        let mut only_n1: [Option<Vec3>; RING_ATOMS] = [None; RING_ATOMS];
        only_n1[0] = Some(Vec3::ZERO);
        assert!(Base::read(&only_n1).is_none());

        // Has the five-ring, so it is read as a purine, but the six-ring it
        // fuses to is missing.
        let mut half: [Option<Vec3>; RING_ATOMS] = [None; RING_ATOMS];
        for slot in [6, 7, 8] {
            half[slot] = Some(Vec3::X);
        }
        assert!(Base::read(&half).is_none());
    }

    /// A flat outline swept along its own normal is a prism, and its faces must
    /// point outwards. A profile wound the wrong way turns the solid inside out,
    /// which the moment pathway reads as negative thickness.
    #[test]
    fn a_ring_is_wound_outwards() {
        let (positions, residues, atoms, names) = nucleotides(3);
        let backbone = Backbone {
            positions: &positions,
            residue_of_atom: &residues,
            name_of_atom: &atoms,
            names: &names,
            sse: &[],
            chain_of_residue: &[],
        };
        let ribbon = build(&backbone, &Style::default());

        // Every face of a closed, outward-wound solid has its normal pointing
        // away from the solid's own centre on average. Summing the signed volume
        // contribution of each triangle is the cheap total form of that: it comes
        // out positive for outward winding and negative for inward.
        let volume: f32 = ribbon
            .indices
            .chunks_exact(3)
            .map(|triangle| {
                let corner = |at: usize| Vec3::from(ribbon.positions[triangle[at] as usize]);
                let (a, b, c) = (corner(0), corner(1), corner(2));
                a.dot(b.cross(c)) / 6.0
            })
            .sum();
        assert!(volume > 0.0, "the mesh is inside out: volume {volume}");
    }

    /// Tubular helices are a different profile, not a different curve, so the
    /// switch must change the vertex count and nothing about the path.
    #[test]
    fn tubular_helices_change_the_profile() {
        let flat = build_helix(8, &[1; 8], &Style::default());
        let tube = build_helix(
            8,
            &[1; 8],
            &Style {
                tubular_helices: true,
                ..Style::default()
            },
        );
        assert!(!tube.is_empty());
        // Both are ellipses of the same side count, so the counts match and the
        // extents do not.
        assert_eq!(flat.positions.len(), tube.positions.len());
        let extent = |ribbon: &Ribbon| {
            ribbon
                .positions
                .iter()
                .map(|p| Vec3::from(*p).length())
                .fold(0.0f32, f32::max)
        };
        assert!(
            extent(&tube) > extent(&flat),
            "a helix tube is fatter than a helix ribbon"
        );
    }
}
