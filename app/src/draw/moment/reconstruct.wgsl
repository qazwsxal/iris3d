// Moments back to a depth-dependent absorbed fraction.
//
// Algorithm 3 ("Hamburger 4MSM") of Peters and Klein, Moment Shadow Mapping,
// I3D 2015 — the same reconstruction MBOIT uses for power moments.
//
// **Nothing imports this yet.** The fullscreen resolve does not need it: the
// accumulation clamps every interval to the opaque depth, so the fraction in
// front of that depth is exactly one and the total is already the right answer.
// This earns its place when something asks at an *intermediate* depth, which is
// the second geometry pass of §2 — transparent actors attenuating themselves.
//
// Before that arrives, read `validate::absorbed_fraction`, which is this
// function in Rust with tests on it. It is not encouraging: four power moments
// bound a *continuous* measure poorly in its interior. A slab of uniform
// absorbance queried at its own midpoint reconstructs at 0.26 against a true
// 0.50. Being a lower bound the error is always towards too little absorbance,
// so volumes read too bright rather than too dark, but it is much larger than
// the reference document's §3.3 suggests when it calls uniform interior
// absorbance "easier to reconstruct than surfaces". It is easier to
// *accumulate*; the reconstruction is what suffers, because the bound is exact
// only for measures living on two points and a slab is the opposite of that.

#define_import_path iris3d::moment_reconstruct

// Interpolation weight towards the fixed vector below.
//
// Without it the Hankel matrix loses positive definiteness and the Cholesky
// decomposition fails, which shows on screen as elongated bands of broken
// pixels rather than as noise. Peters and Klein use 3e-5 for moments quantised
// to 16 bits; these are full f32, so this is more headroom than needed.
const BIAS: f32 = 3.0e-5;

// The moment vector biased towards: half the measure at each end of the domain,
// whose normalised power moments are all one half.
const BIAS_TARGET: vec4<f32> = vec4(0.5, 0.5, 0.5, 0.5);

// The fraction of the absorbance lying in front of `at`, given the normalised
// moments `raw` of the absorbance measure over a `[0, 1]` domain.
//
// A lower bound rather than an estimate, so the error is always towards too
// little absorbance.
fn absorbed_fraction(raw: vec4<f32>, at: f32) -> f32 {
    let b = mix(raw, BIAS_TARGET, BIAS);

    // Solve B c = (1, at, at^2) where B is the Hankel matrix
    //
    //   [ 1   b1  b2 ]
    //   [ b1  b2  b3 ]
    //   [ b2  b3  b4 ]
    //
    // by LDL^T, which stays well behaved even when B is nearly singular.
    let l21 = b.x;
    let l31 = b.y;
    let d2 = b.y - b.x * b.x;
    if d2 <= 0.0 {
        // Degenerate even after biasing: the whole measure sits at one depth.
        // Either that depth is in front of the one being asked about or it is
        // not, with nothing in between to interpolate.
        return select(0.0, 1.0, at > b.x);
    }
    let l32 = (b.z - b.x * b.y) / d2;
    let d3 = b.w - l31 * l31 - l32 * l32 * d2;
    if d3 <= 0.0 {
        return select(0.0, 1.0, at > b.x);
    }

    // Forward substitution, the diagonal, then back substitution.
    let y2 = at - l21;
    let y3 = at * at - l31 - l32 * y2;
    let c3 = y3 / d3;
    let c2 = y2 / d2 - l32 * c3;
    let c1 = 1.0 - l21 * c2 - l31 * c3;

    // The two depths the reconstructed measure is supported on.
    let discriminant = c2 * c2 - 4.0 * c3 * c1;
    if c3 == 0.0 || discriminant < 0.0 {
        return select(0.0, 1.0, at > b.x);
    }
    let root = sqrt(discriminant);
    // Numerically stable quadratic: forming both roots from the same
    // subtraction loses all precision when one of them is small.
    let q = -0.5 * (c2 + sign(c2) * root);
    let first = min(q / c3, c1 / q);
    let second = max(q / c3, c1 / q);

    if at <= first {
        return 0.0;
    }
    if at <= second {
        let numerator = at * second - b.x * (at + second) + b.y;
        return numerator / ((second - first) * (at - first));
    }
    let numerator = first * second - b.x * (first + second) + b.y;
    return 1.0 - numerator / ((at - first) * (at - second));
}
