//! Glycans, drawn as SNFG shapes.
//!
//! The Symbol Nomenclature for Glycans is what the field reads: a blue square is
//! GlcNAc, a green circle is mannose, a yellow circle is galactose, and the
//! shape-and-colour pair identifies the sugar without a label. Drawn in 3D it
//! keeps every symbol at the residue's real position, so the tree is legible
//! *and* correctly placed against the protein it hangs off.
//!
//! Ball-and-stick would be honest and much less legible — a glycan is a dozen
//! near-identical rings, and telling GlcNAc from galactose by eye in a stick
//! model is not something anyone does.
//!
//! # Why the renderer holds the convention
//!
//! The wire carries *which sugar* a residue is, not what to draw for it, exactly
//! as it carries a secondary-structure code rather than "draw a ribbon here".
//! SNFG is a rendering convention, so the shape and the colour belong on this
//! side; a client that wanted different symbols would be disagreeing with the
//! standard, which is not something the format should make easy.
//!
//! # It emits its own colour, and that is the exception
//!
//! Every other mesh-producing filter hands out a scalar and lets `colormap`
//! decide what it looks like. Here the colour *is* half the reading — a blue
//! square is GlcNAc and a yellow square is GalNAc — so it is not a choice to
//! offer downstream. The `colour` output goes straight into
//! [`geometry`](super::geometry). `residue_index` comes out beside it anyway,
//! because a pick has to walk back to a residue whatever the colouring was.
//!
//! # Every shape is a sweep
//!
//! All four solids are one profile swept along an axis with its radius varying,
//! which is what [`cartoon::sweep_run`] already does:
//!
//! - **sphere** — a circle whose radius follows a semicircle, which is a surface
//!   of revolution and so an ordinary UV sphere;
//! - **cube** — a square, constant, two samples;
//! - **diamond** — a square going nothing → full → nothing, an octahedron;
//! - **cone** — a circle going full → nothing.
//!
//! So there is no primitive library here and, more to the point, every shape
//! comes out closed and correctly wound for free — which the moment pathway in
//! [`draw::default`](crate::draw::default) requires.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::data::array::Dtype;
use crate::model::{ParamKind, ParamSpec, float};
use crate::scene::DataArray;

use super::cartoon::{self, Profile, Ribbon, Sample};
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
    ParamSpec {
        id: "residue_snfg",
        label: "sugar per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: true,
            structural: true,
        },
    },
    // Which atoms to build from, by index. Unbound uses all of them.
    //
    // Narrowing belongs here for the reason it does on `cartoon`: a symbol is
    // one closed solid per residue, so cutting the *vertices* of a finished one
    // would cut it open. "Only this chain's sugars" means placing fewer symbols,
    // not hiding part of each.
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
        id: "size",
        label: "symbol size (Å)",
        kind: ParamKind::Float {
            default: 1.6,
            min: 0.3,
            max: 5.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "link_radius",
        label: "link radius (Å)",
        kind: ParamKind::Float {
            default: 0.35,
            min: 0.05,
            max: 1.5,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "link_distance",
        label: "link distance (Å)",
        kind: ParamKind::Float {
            default: 7.0,
            min: 3.0,
            max: 12.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "radial_segments",
        label: "sides of a round symbol",
        kind: ParamKind::Float {
            default: 16.0,
            min: 4.0,
            max: 32.0,
            logarithmic: false,
        },
    },
];

/// Arrays: the four `cartoon` emits, plus the colour.
///
/// The assembly into one mesh is [`geometry`](super::geometry)'s job here as
/// everywhere else, even though nothing downstream will recolour this one: two
/// actors over one glycan still have to share the vertex buffers.
const OUTPUTS: &[OutputSpec] = &[
    OutputSpec {
        id: "positions",
        label: "positions",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Float32),
            shape: &[0, 3],
        },
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
    // Linear RGB per vertex, ready for `geometry`. The one filter that maps its
    // own colours, because SNFG's palette is the notation rather than a choice.
    OutputSpec {
        id: "colour",
        label: "SNFG colour",
        kind: OutputKind::Array {
            dtype: Some(Dtype::Float32),
            shape: &[0, 3],
        },
        provenance: Provenance::Map {
            via: "residue_index",
            of: "residue_index",
        },
    },
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
        id: "glycan",
        label: "glycan (SNFG)",
        params: PARAMS,
        outputs: OUTPUTS,
        run: Some(run),
    });
}

fn run(request: &Request) -> Outcome {
    let Some(input) = read(request) else {
        return Outcome::refused(
            "could not read its atoms: one of positions, residue_index or \
             residue_snfg is unbound or the wrong length",
        );
    };

    let style = Style {
        size: float(&request.params, "size", 1.6),
        link_radius: float(&request.params, "link_radius", 0.35),
        link_distance: float(&request.params, "link_distance", 7.0),
        radial_segments: float(&request.params, "radial_segments", 16.0).round() as usize,
    };

    let (symbols, colours) = build(&input, &style);
    if symbols.is_empty() {
        // Said rather than left silent: a glycan that draws nothing for a
        // protein with no sugars is correct, and one that draws nothing because
        // `residue_snfg` was bound to the wrong array is not, and on screen the
        // two look the same.
        return Outcome::refused("found no sugar residues in the atoms it was given");
    }

    debug!(
        "glycan: {} vertices, {} triangles",
        symbols.positions.len(),
        symbols.indices.len() / 3
    );

    let vertices = symbols.positions.len() as u64;
    let mut products = Products::new();
    products.insert(
        "positions",
        DataArray::numeric(
            Dtype::Float32,
            vec![vertices, 3],
            floats(&symbols.positions),
        )
        .into(),
    );
    products.insert(
        "normals",
        DataArray::numeric(Dtype::Float32, vec![vertices, 3], floats(&symbols.normals)).into(),
    );
    products.insert(
        "indices",
        DataArray::numeric(
            Dtype::Uint32,
            vec![symbols.indices.len() as u64 / 3, 3],
            symbols
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect(),
        )
        .into(),
    );
    products.insert(
        "colour",
        DataArray::numeric(Dtype::Float32, vec![vertices, 3], floats(&colours)).into(),
    );
    products.insert(
        "residue_index",
        DataArray::numeric(
            Dtype::Uint32,
            vec![vertices],
            symbols
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

/// How big a symbol is and how finely it is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style {
    /// Half the width of a symbol, in ångströms.
    pub size: f32,
    /// Radius of the stick joining two linked sugars.
    pub link_radius: f32,
    /// Sides of a rounded symbol.
    pub radial_segments: usize,
    /// How far apart two ring centroids may be and still be called linked.
    pub link_distance: f32,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            // SNFG symbols are drawn all one size — the shape carries the
            // identity, so scaling them by anything would be saying something
            // the standard does not say.
            size: 1.6,
            link_radius: 0.35,
            radial_segments: 16,
            // Two sugars joined by a glycosidic bond sit about 5.5 A apart
            // centroid to centroid; 7 admits the longer linkages without
            // reaching a neighbouring branch.
            link_distance: 7.0,
        }
    }
}

/// One SNFG symbol: a shape and a colour.
///
/// The pair is the identity — a blue square is GlcNAc and a blue circle is
/// glucose, and neither half means anything alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symbol {
    pub shape: Shape,
    pub colour: Tint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// A filled circle in the 2D notation; a sphere here.
    Circle,
    Square,
    /// A filled diamond; an octahedron here.
    Diamond,
    /// A filled triangle; a cone here.
    Triangle,
}

/// SNFG's fixed palette. Quoted in sRGB, as every colour in this project is.
///
/// The whole ten, including the ones no sugar in [`symbol`] uses yet. The
/// palette is the standard's and is not ours to trim: a colour with no sugar
/// against it today is a sugar not yet in the table, not a colour that should
/// be deleted.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tint {
    White,
    Blue,
    Green,
    Yellow,
    Cyan,
    Pink,
    Purple,
    Brown,
    Orange,
    Red,
}

impl Tint {
    /// Linear RGB, because vertex colours reach the shader unconverted.
    ///
    /// Three channels rather than four: `geometry` gives every vertex an opaque
    /// alpha, and a glycan symbol is a notation rather than something to see
    /// through.
    fn linear(self) -> [f32; 3] {
        let rgb = match self {
            Tint::White => [1.00, 1.00, 1.00],
            Tint::Blue => [0.00, 0.56, 0.84],
            Tint::Green => [0.00, 0.64, 0.31],
            Tint::Yellow => [1.00, 0.83, 0.00],
            Tint::Cyan => [0.55, 0.85, 0.94],
            Tint::Pink => [0.96, 0.60, 0.75],
            Tint::Purple => [0.64, 0.31, 0.64],
            Tint::Brown => [0.58, 0.35, 0.16],
            Tint::Orange => [0.96, 0.51, 0.15],
            Tint::Red => [0.65, 0.00, 0.15],
        };
        Color::srgb(rgb[0], rgb[1], rgb[2])
            .to_linear()
            .to_f32_array()[..3]
            .try_into()
            .expect("three of the four channels")
    }
}

/// The sugar codes the wire carries, and the symbol each one draws.
///
/// The codes are iris3d's own, assigned in `molecules.py`; 0 means "not a
/// sugar" and is the value every non-glycan residue has. An unrecognised sugar
/// arrives as [`UNKNOWN`] and draws a white circle, which is SNFG's own answer
/// for an unassigned monosaccharide rather than a placeholder invented here.
pub const UNKNOWN: u8 = 1;

fn symbol(code: u8) -> Option<Symbol> {
    let (shape, colour) = match code {
        UNKNOWN => (Shape::Circle, Tint::White),
        2 => (Shape::Circle, Tint::Blue),     // glucose
        3 => (Shape::Square, Tint::Blue),     // GlcNAc
        4 => (Shape::Circle, Tint::Green),    // mannose
        5 => (Shape::Circle, Tint::Yellow),   // galactose
        6 => (Shape::Square, Tint::Yellow),   // GalNAc
        7 => (Shape::Triangle, Tint::Red),    // fucose
        8 => (Shape::Diamond, Tint::Purple),  // Neu5Ac, sialic acid
        9 => (Shape::Diamond, Tint::Cyan),    // Neu5Gc
        10 => (Shape::Square, Tint::Orange),  // xylose
        11 => (Shape::Diamond, Tint::Blue),   // glucuronic acid
        12 => (Shape::Diamond, Tint::Orange), // iduronic acid
        13 => (Shape::Circle, Tint::Pink),    // rhamnose
        14 => (Shape::Square, Tint::Green),   // ManNAc
        _ => return None,
    };
    Some(Symbol { shape, colour })
}

/// One sugar residue, reduced to what drawing it needs.
#[derive(Debug, Clone, Copy)]
struct Sugar {
    residue: u32,
    centre: Vec3,
    /// The ring plane's normal, which orients a square or a triangle so it does
    /// not read as an arbitrary box.
    normal: Vec3,
    symbol: Symbol,
}

/// The arrays a glycan reads, already decoded.
pub struct Input {
    positions: Vec<Vec3>,
    residue_of_atom: Vec<u32>,
    snfg: Vec<u8>,
}

/// Reads what was bound.
///
/// Only three arrays, and none of them is the atom-name column the cartoon
/// needs: a symbol sits at the residue's centroid, so which atom is which does
/// not come into it.
///
/// `atoms` narrows the atoms before any of that, so a residue left with fewer
/// than three of them stops being drawn rather than being placed off a worse
/// plane.
fn read(request: &Request) -> Option<Input> {
    let all_positions = request.input("positions")?.to_vec3();
    let all_residues = request.input("residue_index")?.to_u32()?;
    // A `uint8` array's bytes are its values, so the codes need no decode.
    let snfg = request.input("residue_snfg")?.data.clone();
    if all_positions.is_empty() || all_residues.len() < all_positions.len() {
        return None;
    }

    let kept = request.input("atoms").and_then(|array| array.to_u32());
    let (positions, residue_of_atom) = match &kept {
        Some(kept) => (
            kept.iter()
                .filter_map(|index| all_positions.get(*index as usize).copied())
                .collect(),
            kept.iter()
                .filter_map(|index| all_residues.get(*index as usize).copied())
                .collect(),
        ),
        None => (all_positions, all_residues),
    };

    Some(Input {
        positions,
        residue_of_atom,
        snfg,
    })
}

/// Builds the SNFG symbols and the sticks between them.
pub fn build(input: &Input, style: &Style) -> (Ribbon, Vec<[f32; 3]>) {
    let sugars = sugars(input);
    let mut mesh = Ribbon::default();
    let mut colours: Vec<[f32; 3]> = Vec::new();

    for sugar in &sugars {
        let before = mesh.positions.len();
        sweep_symbol(sugar, style, &mut mesh);
        colours.resize(mesh.positions.len(), sugar.symbol.colour.linear());
        debug_assert!(mesh.positions.len() >= before);
    }

    // Linkages. Distance between ring centroids rather than the bond list,
    // which keeps this to three bound arrays — and a glycan's connectivity is
    // unambiguous at this scale, because two sugars that close are joined.
    // The cost is honest: a structure with two glycans packed against each
    // other could draw a stick that is not a bond.
    let limit = style.link_distance * style.link_distance;
    for (first, sugar) in sugars.iter().enumerate() {
        for other in &sugars[first + 1..] {
            if sugar.centre.distance_squared(other.centre) > limit {
                continue;
            }
            let before = mesh.positions.len();
            sweep_link(sugar, other, style, &mut mesh);
            // Half each, so a link reads as belonging to both ends.
            let added = mesh.positions.len() - before;
            let half = before + added / 2;
            colours.resize(half, sugar.symbol.colour.linear());
            colours.resize(mesh.positions.len(), other.symbol.colour.linear());
        }
    }

    (mesh, colours)
}

/// One sugar per residue that has a symbol and enough atoms to place it.
fn sugars(input: &Input) -> Vec<Sugar> {
    let mut atoms: HashMap<u32, Vec<Vec3>> = HashMap::default();
    let count = input.positions.len().min(input.residue_of_atom.len());
    for atom in 0..count {
        let residue = input.residue_of_atom[atom];
        let Some(code) = input.snfg.get(residue as usize).copied() else {
            continue;
        };
        if code == 0 {
            continue;
        }
        atoms
            .entry(residue)
            .or_default()
            .push(input.positions[atom]);
    }

    let mut sugars: Vec<Sugar> = atoms
        .into_iter()
        .filter_map(|(residue, positions)| {
            // Three atoms is the least that defines a plane; fewer is a residue
            // so poorly resolved that a symbol would be a guess.
            if positions.len() < 3 {
                return None;
            }
            let symbol = symbol(input.snfg[residue as usize])?;
            let centre = positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
            // Newell's over the atoms in file order. A sugar ring's atoms are
            // deposited in ring order, so this is the ring plane; for a residue
            // whose atoms are not, it is still a stable plane through them,
            // which is all a symbol's orientation needs.
            let mut normal = Vec3::ZERO;
            for index in 0..positions.len() {
                let (from, to) = (positions[index], positions[(index + 1) % positions.len()]);
                normal += (from - centre).cross(to - centre);
            }
            Some(Sugar {
                residue,
                centre,
                normal: normal.normalize_or(Vec3::Y),
                symbol,
            })
        })
        .collect();
    // Residue order, so the mesh is built the same way twice for the same input.
    sugars.sort_unstable_by_key(|sugar| sugar.residue);
    sugars
}

/// Sweeps one symbol as a closed solid.
fn sweep_symbol(sugar: &Sugar, style: &Style, mesh: &mut Ribbon) {
    let axis = sugar.normal;
    let across = axis.any_orthonormal_vector();
    let up = axis.cross(across).normalize_or(Vec3::Y);
    let size = style.size.max(0.01);
    let sides = style.radial_segments.clamp(3, 64);

    let frame = |offset: f32, radius: f32| {
        Sample::frame(
            sugar.centre + axis * offset,
            axis,
            across,
            up,
            sugar.residue,
            radius,
            radius,
        )
    };

    match sugar.symbol.shape {
        // A surface of revolution: the radius follows a semicircle, which is
        // exactly a UV sphere. The poles are zero-radius rings, so their caps
        // are degenerate triangles that cost nothing and cover nothing.
        Shape::Circle => {
            let rings = sides.max(6) / 2;
            let samples: Vec<Sample> = (0..=rings)
                .map(|ring| {
                    let angle = std::f32::consts::PI * ring as f32 / rings as f32;
                    frame(-size * angle.cos(), size * angle.sin())
                })
                .collect();
            cartoon::sweep_run(&samples, &Profile::rounded(sides), mesh);
        }
        Shape::Square => {
            cartoon::sweep_run(
                &[frame(-size, size), frame(size, size)],
                &Profile::rectangular(),
                mesh,
            );
        }
        // Nothing, full, nothing: an octahedron.
        Shape::Diamond => {
            cartoon::sweep_run(
                &[frame(-size, 0.0), frame(0.0, size), frame(size, 0.0)],
                &Profile::rectangular(),
                mesh,
            );
        }
        Shape::Triangle => {
            cartoon::sweep_run(
                &[frame(-size, size), frame(size, 0.0)],
                &Profile::rounded(sides),
                mesh,
            );
        }
    }
}

/// A stick between two linked sugars.
fn sweep_link(from: &Sugar, to: &Sugar, style: &Style, mesh: &mut Ribbon) {
    let along = to.centre - from.centre;
    let length = along.length();
    if length < f32::EPSILON {
        return;
    }
    let axis = along / length;
    let across = axis.any_orthonormal_vector();
    let up = axis.cross(across).normalize_or(Vec3::Y);
    let radius = style.link_radius.max(0.01);
    let middle = (from.centre + to.centre) * 0.5;

    // Two runs, so each half can take its own end's colour.
    for (start, end, residue) in [
        (from.centre, middle, from.residue),
        (middle, to.centre, to.residue),
    ] {
        cartoon::sweep_run(
            &[
                Sample::frame(start, axis, across, up, residue, radius, radius),
                Sample::frame(end, axis, across, up, residue, radius, radius),
            ],
            &Profile::rounded(style.radial_segments.clamp(3, 64)),
            mesh,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ParamMap, ParamValue};

    /// A ring of six atoms in the xy plane, at `centre`.
    fn ring(centre: Vec3) -> Vec<Vec3> {
        (0..6)
            .map(|corner| {
                let angle = std::f32::consts::TAU * corner as f32 / 6.0;
                centre + Vec3::new(1.4 * angle.cos(), 1.4 * angle.sin(), 0.0)
            })
            .collect()
    }

    /// Two linked sugars: a GlcNAc and a mannose, 5.5 A apart.
    fn input() -> Input {
        let mut positions = Vec::new();
        let mut residue_of_atom = Vec::new();
        for (residue, centre) in [Vec3::ZERO, Vec3::new(5.5, 0.0, 0.0)].iter().enumerate() {
            for atom in ring(*centre) {
                positions.push(atom);
                residue_of_atom.push(residue as u32);
            }
        }
        Input {
            positions,
            residue_of_atom,
            snfg: vec![3, 4],
        }
    }

    /// The same sugars, as the arrays a client would bind.
    fn request(input: &Input) -> Request {
        let mut inputs = HashMap::new();
        inputs.insert(
            "positions",
            DataArray::numeric(
                Dtype::Float32,
                vec![input.positions.len() as u64, 3],
                input
                    .positions
                    .iter()
                    .flat_map(|p| [p.x, p.y, p.z])
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
        );
        inputs.insert(
            "residue_index",
            DataArray::numeric(
                Dtype::Uint32,
                vec![input.residue_of_atom.len() as u64],
                input
                    .residue_of_atom
                    .iter()
                    .flat_map(|r| r.to_le_bytes())
                    .collect(),
            ),
        );
        inputs.insert(
            "residue_snfg",
            DataArray::numeric(
                Dtype::Uint8,
                vec![input.snfg.len() as u64],
                input.snfg.clone(),
            ),
        );
        Request {
            params: ParamMap::default(),
            inputs,
        }
    }

    /// Every symbol and every stick is a closed solid, welded by position — the
    /// moment pathway draws an open one too clear rather than failing.
    #[test]
    fn every_symbol_is_closed() {
        let (mesh, colours) = build(&input(), &Style::default());
        assert!(!mesh.is_empty());
        assert_eq!(colours.len(), mesh.positions.len());

        let key = |index: u32| {
            let [x, y, z] = mesh.positions[index as usize];
            let grid = |v: f32| (v * 10_000.0).round() as i64;
            (grid(x), grid(y), grid(z))
        };
        let mut welded: HashMap<(i64, i64, i64), u32> = HashMap::default();
        let mut canonical = Vec::with_capacity(mesh.positions.len());
        for index in 0..mesh.positions.len() as u32 {
            let next = welded.len() as u32;
            canonical.push(*welded.entry(key(index)).or_insert(next));
        }
        let mut edges: HashMap<(u32, u32), i32> = HashMap::default();
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [
                canonical[triangle[0] as usize],
                canonical[triangle[1] as usize],
                canonical[triangle[2] as usize],
            ];
            if a == b || b == c || a == c {
                continue;
            }
            for (start, end) in [(a, b), (b, c), (c, a)] {
                *edges.entry((start.min(end), start.max(end))).or_default() +=
                    if start < end { 1 } else { -1 };
            }
        }
        let open = edges.values().filter(|count| **count != 0).count();
        assert_eq!(open, 0, "{open} unpaired edges");
    }

    /// The two sugars get different colours, which is the whole point of SNFG:
    /// the symbol identifies the residue without a label.
    #[test]
    fn each_sugar_takes_its_own_colour() {
        let (_, colours) = build(&input(), &Style::default());
        let distinct: Vec<[f32; 3]> = {
            let mut seen: Vec<[f32; 3]> = Vec::new();
            for colour in &colours {
                if !seen.contains(colour) {
                    seen.push(*colour);
                }
            }
            seen
        };
        assert_eq!(distinct.len(), 2, "a GlcNAc and a mannose are two colours");
        assert!(distinct.contains(&Tint::Blue.linear()));
        assert!(distinct.contains(&Tint::Green.linear()));
    }

    /// Residues that are not sugars draw nothing at all, which is what lets a
    /// glycan read the same arrays as a cartoon over a whole structure.
    #[test]
    fn non_sugars_draw_nothing() {
        let mut input = input();
        input.snfg = vec![0, 0];
        let (mesh, colours) = build(&input, &Style::default());
        assert!(mesh.is_empty());
        assert!(colours.is_empty());
    }

    /// Sugars too far apart to be bonded get no stick between them.
    #[test]
    fn distant_sugars_are_not_linked() {
        let mut far = input();
        for atom in 6..12 {
            far.positions[atom] += Vec3::new(40.0, 0.0, 0.0);
        }
        let (linked, _) = build(&input(), &Style::default());
        let (apart, _) = build(&far, &Style::default());
        assert!(
            apart.positions.len() < linked.positions.len(),
            "a stick should have been left out"
        );
    }

    /// A residue with fewer than three atoms cannot define a plane, so it is
    /// skipped rather than drawn with a guessed orientation.
    #[test]
    fn a_barely_resolved_sugar_is_skipped() {
        let input = Input {
            positions: vec![Vec3::ZERO, Vec3::X],
            residue_of_atom: vec![0, 0],
            snfg: vec![3],
        };
        assert!(build(&input, &Style::default()).0.is_empty());
    }

    /// The outputs line up: one colour and one residue per vertex, three
    /// indices per triangle. A consumer binds `colour` straight into
    /// `geometry`, which drops a colour array whose length disagrees with the
    /// positions rather than complaining, so the check belongs here.
    #[test]
    fn the_outputs_are_all_per_vertex() {
        let outcome = run(&request(&input()));
        assert!(outcome.problem.is_none(), "{:?}", outcome.problem);
        let array = |id: &str| outcome.products[id].array().expect("an array");
        let vertices = array("positions").shape[0];
        assert_eq!(array("normals").shape[0], vertices);
        assert_eq!(array("colour").shape[0], vertices);
        assert_eq!(array("residue_index").shape[0], vertices);
        assert_eq!(array("indices").shape[1], 3);
    }

    /// `residue_index` is what a pick walks back along, so every vertex has to
    /// name one of the residues that were actually drawn.
    #[test]
    fn every_vertex_names_its_residue() {
        let outcome = run(&request(&input()));
        let residues = outcome.products["residue_index"]
            .array()
            .expect("an array")
            .to_u32()
            .expect("integers");
        assert!(!residues.is_empty());
        assert!(residues.iter().all(|residue| *residue < 2), "{residues:?}");
    }

    /// A structure with no sugars refuses rather than going quiet: an empty
    /// output and a misbound `residue_snfg` look identical on screen.
    #[test]
    fn a_structure_with_no_sugars_says_so() {
        let mut plain = input();
        plain.snfg = vec![0, 0];
        assert!(run(&request(&plain)).is_refusal());
    }

    /// `atoms` narrows the atoms, not the finished vertices: dropping the
    /// second sugar's atoms drops its symbol and the stick with it.
    #[test]
    fn atoms_narrows_which_sugars_are_placed() {
        let mut request = request(&input());
        request.inputs.insert(
            "atoms",
            DataArray::numeric(
                Dtype::Uint32,
                vec![6],
                (0u32..6).flat_map(|i| i.to_le_bytes()).collect(),
            ),
        );
        let residues = run(&request).products["residue_index"]
            .array()
            .expect("an array")
            .to_u32()
            .expect("integers");
        assert!(residues.iter().all(|residue| *residue == 0), "{residues:?}");
    }

    #[test]
    fn params_are_declared_for_every_input_the_run_reads() {
        for id in ["positions", "residue_index", "residue_snfg", "atoms"] {
            assert!(
                PARAMS.iter().any(|spec| spec.id == id),
                "the run reads \"{id}\" but no parameter declares it"
            );
        }
    }

    /// The settings survive normalisation, so a size dragged in the interface
    /// reaches the run rather than being dropped as an unknown key.
    #[test]
    fn the_kind_normalises_its_settings() {
        let mut registry = FilterRegistry::default();
        register(&mut registry);
        let kind = registry.get("glycan").expect("just registered");

        let normalised = kind.normalise(&ParamMap::default());
        assert_eq!(normalised.get("size"), Some(&ParamValue::Float(1.6)));
        assert_eq!(
            normalised.get("radial_segments"),
            Some(&ParamValue::Float(16.0))
        );
    }
}
