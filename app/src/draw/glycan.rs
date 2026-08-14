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
//! comes out closed and correctly wound for free — which
//! [`default`](super::default) requires.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::filter::cartoon::{self, Profile, Ribbon, Sample};
use crate::scene::registry::Bindings;
use crate::scene::{DataArray, DataStore, Subset};

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
    /// Linear RGBA, because vertex colours reach the shader unconverted.
    fn linear(self) -> [f32; 4] {
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
            .to_f32_array()
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

/// Reads what an actor bound, narrowed to its subset.
///
/// Only three arrays, and none of them is the atom-name column the cartoon
/// needs: a symbol sits at the residue's centroid, so which atom is which does
/// not come into it.
pub fn read(
    bindings: &Bindings,
    subset: &Subset,
    store: &DataStore,
    arrays: &Assets<DataArray>,
) -> Option<Input> {
    let positions = super::bound(bindings, "positions", store, arrays)?;
    let all_positions = positions.to_vec3();
    let all_residues = super::bound(bindings, "residue_index", store, arrays)?.to_u32()?;
    let snfg = super::bound(bindings, "residue_snfg", store, arrays)?;
    if all_positions.is_empty() || all_residues.len() < all_positions.len() {
        return None;
    }

    let kept = subset.selected(all_positions.len(), arrays);
    Some(Input {
        positions: match &kept {
            Some(kept) => kept
                .iter()
                .filter_map(|index| all_positions.get(*index as usize).copied())
                .collect(),
            None => all_positions,
        },
        residue_of_atom: match &kept {
            Some(kept) => kept
                .iter()
                .filter_map(|index| all_residues.get(*index as usize).copied())
                .collect(),
            None => all_residues,
        },
        // Per residue, and so not narrowed: a subset renumbers atoms, not
        // residues.
        snfg: snfg.data.clone(),
    })
}

/// Builds the SNFG symbols and the sticks between them.
pub fn build(input: &Input, style: &Style) -> (Ribbon, Vec<[f32; 4]>) {
    let sugars = sugars(input);
    let mut mesh = Ribbon::default();
    let mut colours: Vec<[f32; 4]> = Vec::new();

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
        atoms.entry(residue).or_default().push(input.positions[atom]);
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
        let distinct: Vec<[f32; 4]> = {
            let mut seen: Vec<[f32; 4]> = Vec::new();
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
    /// glycan actor bind the same arrays as a cartoon over a whole structure.
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
}
