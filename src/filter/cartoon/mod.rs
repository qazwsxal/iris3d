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

use crate::data::array::Dtype;
use crate::model::{ParamKind, ParamSpec, flag, float};
use crate::scene::DataArray;

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

mod read;
mod spline;
mod style;
mod sweep;

#[cfg(test)]
mod tests;

pub use spline::Sample;
pub use style::Style;
pub use sweep::{Profile, sweep_run};

use read::{orient, read, segments, smooth_strands};
use spline::sample;
use style::Polymer;

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

/// Builds the cartoon for one structure.
pub fn build(backbone: &Backbone, style: &Style) -> Ribbon {
    let mut ribbon = Ribbon::default();
    let (mut residues, mut nucleic, mut bases) = (0usize, 0usize, 0usize);
    for mut segment in segments(backbone) {
        orient(&mut segment);
        smooth_strands(&mut segment);
        let samples = sample(&segment, style);
        sweep::sweep(&samples, style, &mut ribbon);

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
                    sweep::sweep_base(&base, node.position, node.residue, style, &mut ribbon);
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
