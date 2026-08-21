//! What a cartoon looks like, and what the residue codes mean.
//!
//! [`Style`] is the tunable half — widths, thicknesses, how finely the curve is
//! sampled. The rest is the fixed half: which secondary-structure code means
//! which [`Form`], which atom name is a guide for which [`Polymer`], and where a
//! nucleic base's ring atoms sit. Those are facts about the file formats and
//! about chemistry, not settings.

use super::*;

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

/// What a stretch of backbone is drawn as.
///
/// Fewer states than the wire carries, because this is about geometry: a 3-10
/// helix and a pi helix are drawn as helices, and nothing distinguishes a turn
/// from a bend once both are tubes. The eight-state code is still what travels,
/// so a later renderer that *does* want them apart loses nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Form {
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
    pub(super) fn of_code(code: u8) -> Self {
        match code {
            1 | 4 | 5 => Form::Helix,
            2 | 3 => Form::Strand,
            _ => Form::Coil,
        }
    }

    /// Half-width across the ribbon and half-thickness through it.
    pub(super) fn size(self, style: &Style) -> (f32, f32) {
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
    pub(super) fn profile(self, style: &Style) -> Profile {
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
pub(super) enum Polymer {
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
    pub(super) fn gap(self) -> f32 {
        match self {
            Polymer::Protein => 5.0,
            Polymer::Nucleic => 9.0,
        }
    }
}

/// Which of a residue's two guide atoms a name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Guide {
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
pub(super) fn guide_role(name: &str) -> Option<(Polymer, Guide, u8)> {
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

/// Which of the base-ring atoms a name is, in the order `Base` reads them:
/// N1, C2, N3, C4, C5, C6, N7, C8, N9.
///
/// `C2` is also the nucleic direction atom. Both lookups run over every atom, so
/// it simply fills two roles rather than needing a rule about which wins.
pub(super) fn base_slot(name: &str) -> Option<usize> {
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
