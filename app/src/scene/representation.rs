//! How an object gets drawn.
//!
//! Representations are **child entities** of the scene object, so one dataset
//! can carry several at once — a protein as cartoon *and* licorice, a grid as
//! outline *and* volume. That also makes it possible to run two rendering
//! approaches over the same data side by side and compare them, which is a
//! goal rather than an accident.
//!
//! Nothing here draws anything. A rendering backend is a plugin that queries
//! `(&Representation, &ChildOf)`, reads the parent's dataset, and produces
//! whatever it produces. [`crate::draw`] is the current one — a straightforward
//! `Mesh3d`-per-representation baseline — and the split exists so a second can
//! run beside it rather than replace it.

use bevy::prelude::*;

use super::dataset::DatasetKind;

/// A way of drawing the parent object.
// Wireframe, Isosurface, Glyphs, Cartoon and SpaceFilling are declared but no
// backend constructs them; `unimplemented_for` exists to show them in the UI as
// unavailable. Both they and this allow go when representation kinds move to a
// registry that backends populate, at which point a kind exists iff something
// can draw it.
#[allow(dead_code)]
#[derive(Component, Debug, Clone, PartialEq)]
pub enum Representation {
    /// Points drawn as camera-facing discs. `size` is the diameter in world
    /// units, so a sensible value depends on the data's own scale — there is
    /// no universally right default until a client can set it.
    Points { size: f32 },
    /// Cell surfaces, shaded.
    Surface,
    /// Cell edges only.
    Wireframe,
    /// A level set extracted from a scalar field.
    Isosurface { field: String, value: f32 },
    /// Oriented glyphs driven by a vector or tensor field — arrows for vectors,
    /// ellipsoids for stress and strain.
    Glyphs {
        field: String,
        glyph: GlyphKind,
        scale: f32,
    },
    /// Direct volume rendering of a scalar field over a grid.
    VolumeRender { field: String },

    // Molecular idioms. Distinct variants rather than parameters on Surface,
    // because their geometry is generated from residue and bond structure
    // rather than from cells.
    /// Secondary-structure ribbon. Needs residues and chains.
    Cartoon,
    /// Spheres at atoms, cylinders along bonds.
    BallAndStick { atom_scale: f32, bond_radius: f32 },
    /// Van der Waals spheres.
    SpaceFilling,
}

impl Representation {
    /// A reasonable first way to draw each dataset kind, used when an upload
    /// does not ask for anything specific.
    pub fn default_for(kind: DatasetKind) -> Option<Self> {
        match kind {
            DatasetKind::Points => Some(Representation::Points { size: 0.05 }),
            DatasetKind::Mesh => Some(Representation::Surface),
            DatasetKind::Molecule => Some(Representation::BallAndStick {
                atom_scale: 0.25,
                bond_radius: 0.1,
            }),
            DatasetKind::Grid => Some(Representation::VolumeRender {
                field: String::new(),
            }),
            // Nothing sensible to draw without knowing what the arrays mean,
            // and nothing at all to draw for a grouping node.
            DatasetKind::Raw | DatasetKind::Empty => None,
        }
    }

    /// Representations a backend can actually draw for this dataset kind.
    ///
    /// Only implemented variants are listed. Offering the rest would put
    /// choices in the UI that silently do nothing — see
    /// [`Self::unimplemented_for`], which is how the rest are surfaced.
    pub fn available_for(kind: DatasetKind) -> Vec<Self> {
        match kind {
            DatasetKind::Points => vec![Representation::Points { size: 0.05 }],
            DatasetKind::Mesh => vec![Representation::Surface],
            DatasetKind::Molecule => vec![Representation::BallAndStick {
                atom_scale: 0.25,
                bond_radius: 0.1,
            }],
            DatasetKind::Grid | DatasetKind::Raw | DatasetKind::Empty => Vec::new(),
        }
    }

    /// Variants declared but not yet drawable, listed so the UI can show them
    /// as unavailable rather than pretend they do not exist.
    pub fn unimplemented_for(kind: DatasetKind) -> &'static [&'static str] {
        match kind {
            DatasetKind::Points => &["glyphs"],
            DatasetKind::Mesh => &["wireframe", "isosurface", "glyphs"],
            DatasetKind::Molecule => &["cartoon", "space-filling"],
            DatasetKind::Grid => &["volume", "isosurface"],
            DatasetKind::Raw | DatasetKind::Empty => &[],
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Representation::Points { .. } => "points",
            Representation::Surface => "surface",
            Representation::Wireframe => "wireframe",
            Representation::Isosurface { .. } => "isosurface",
            Representation::Glyphs { .. } => "glyphs",
            Representation::VolumeRender { .. } => "volume",
            Representation::Cartoon => "cartoon",
            Representation::BallAndStick { .. } => "ball-and-stick",
            Representation::SpaceFilling => "space-filling",
        }
    }
}

// Reachable only through `Representation::Glyphs`, which nothing constructs.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphKind {
    /// For vector fields.
    Arrow,
    /// Principal axes of a rank-2 tensor.
    Ellipsoid,
    /// Eigenvector directions without magnitude scaling.
    Axes,
}

/// How a representation takes its colour.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct ColorBy {
    /// Field to map. `None` paints a flat colour.
    pub field: Option<String>,
    pub map: ColorMap,
    /// Value range mapped across the colour map. `None` autoscales to the
    /// field's own range.
    pub range: Option<(f32, f32)>,
    /// Used when `field` is `None`.
    pub flat: Color,
}

impl Default for ColorBy {
    fn default() -> Self {
        Self {
            field: None,
            map: ColorMap::Viridis,
            range: None,
            flat: Color::srgb(0.8, 0.8, 0.85),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMap {
    /// Perceptually uniform; a safe default for scalar fields.
    #[default]
    Viridis,
    /// Diverging, for signed quantities about zero.
    CoolWarm,
    Grayscale,
    /// Standard element colouring for molecular data.
    ///
    /// Never selected today: molecules apply CPK colours directly and never
    /// consult the map, so this variant is what that behaviour *should* be
    /// named once element colouring routes through `draw::sample` like every
    /// other map.
    #[allow(dead_code)]
    ByElement,
}
