//! The curve itself: where the samples are, and which way they face.
//!
//! A [`Sample`] is one point along the spline with the frame and the size the
//! sweep needs. Producing them is a cardinal spline through the segment's
//! nodes, with the tension varying by what the residues are and the frame
//! averaged so it turns smoothly rather than flipping.

use super::read::*;
use super::style::*;
use super::*;

/// One point along the curve, with the frame and the size the sweep needs.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    pub(super) position: Vec3,
    pub(super) tangent: Vec3,
    /// Across the ribbon: the wide axis.
    pub(super) across: Vec3,
    /// Through the ribbon: the thin axis, and the flat face's normal.
    pub(super) up: Vec3,
    pub(super) residue: u32,
    pub(super) half_width: f32,
    pub(super) half_thick: f32,
    pub(super) form: Form,
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
pub(super) const ELEMENT_TENSION: f32 = 0.9;
/// Tension at the ends of one, where a looser curve joins the next without a
/// corner. Mol\* pins the boundary to this value; 0.5 is plain Catmull-Rom.
pub(super) const JOIN_TENSION: f32 = 0.5;

/// Samples the spline through a segment, frame and size included.
pub(super) fn sample(segment: &Segment, style: &Style) -> Vec<Sample> {
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
pub(super) fn steady_arrows(samples: &mut [Sample]) {
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
pub(super) fn pair_forms(segment: &Segment) -> Vec<Form> {
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
pub(super) fn tensions(segment: &Segment) -> Vec<f32> {
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
pub(super) fn pair_size(
    segment: &Segment,
    forms: &[Form],
    pair: usize,
    t: f32,
    style: &Style,
) -> (f32, f32) {
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
pub(super) fn frame(
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
pub(super) fn average_frames(samples: &mut [Sample]) {
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
pub(super) fn cardinal(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
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
pub(super) fn tangent(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    pub(super) const DELTA: f32 = 0.01;
    let before = cardinal(p0, p1, p2, p3, t - DELTA, tension);
    let after = cardinal(p0, p1, p2, p3, t + DELTA, tension);
    (after - before).normalize_or(Vec3::Z)
}

/// One point of a closed cross-section, in units of half-width and
/// half-thickness.
#[derive(Debug, Clone, Copy)]
pub(super) struct Rim {
    /// Across the ribbon, in -1..1.
    pub(super) u: f32,
    /// Through the ribbon, in -1..1.
    pub(super) v: f32,
    /// The outward normal in the same two axes, or `None` to take the ellipse
    /// gradient at this point — which is `(u / half_width, v / half_thick)`, and
    /// so cannot be precomputed because it depends on the size at each sample.
    pub(super) normal: Option<Vec2>,
}
