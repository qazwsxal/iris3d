//! Turning samples into triangles.
//!
//! A [`Profile`] is the cross-section swept along the curve. Sweeping it is the
//! last step: every sample contributes one ring of vertices, consecutive rings
//! are stitched, and the ends are capped. Arrowheads and nucleic bases are the
//! two shapes that need more than a plain sweep.

use super::read::*;
use super::spline::*;
use super::style::*;
use super::*;

/// A closed cross-section.
///
/// Two lists rather than one because a hard edge needs a point twice, once per
/// face, and a cap needs it once. `rim` is what the sides are built from and
/// carries the duplicates; `outline` is what a cap is fanned over and does not.
/// Both are wound the same way — counter-clockwise in the across-up plane, which
/// with the sweep advancing along the tangent puts the front faces outward.
pub struct Profile {
    pub(super) rim: Vec<Rim>,
    pub(super) outline: Vec<Vec2>,
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
pub(super) fn sweep(samples: &[Sample], style: &Style, ribbon: &mut Ribbon) {
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
pub(super) fn sweep_arrow(run: &[Sample], profile: &Profile, ribbon: &mut Ribbon) {
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
pub(super) fn cap(ribbon: &mut Ribbon, sample: &Sample, profile: &Profile, front: bool) {
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

pub(super) fn push(ribbon: &mut Ribbon, position: Vec3, normal: Vec3, residue: u32) {
    ribbon.positions.push([position.x, position.y, position.z]);
    ribbon.normals.push([normal.x, normal.y, normal.z]);
    ribbon.residue.push(residue);
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
pub(super) fn sweep_base(
    base: &Base,
    trace: Vec3,
    residue: u32,
    style: &Style,
    ribbon: &mut Ribbon,
) {
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
