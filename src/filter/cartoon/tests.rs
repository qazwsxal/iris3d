//! Tests for the cartoon, across all of its stages.

use super::read::*;
use super::spline::*;
use super::style::*;
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
    let named: [Vec3; RING_ATOMS] = std::array::from_fn(|slot| Vec3::new(slot as f32, 0.0, 0.0));
    let atoms = |five: bool| {
        let mut found: [Option<Vec3>; RING_ATOMS] = std::array::from_fn(|slot| Some(named[slot]));
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
