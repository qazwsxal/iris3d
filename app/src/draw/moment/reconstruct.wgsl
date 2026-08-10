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
// *accumulate*; the reconstruction is what suffers.
//
// # Would trigonometric moments do better? No.
//
// §1 of the reference document says to prefer the trigonometric basis for
// volumetric content, so this was measured rather than assumed. For each basis,
// a linear program over every non-negative measure matching the moments gives
// the tightest achievable bounds on the absorbed fraction — which is basis- and
// algorithm-independent, and so settles the question rather than comparing two
// implementations. At equal cost (a complex moment is two floats, so two
// trigonometric moments against four power moments), on a uniform slab spanning
// most of the domain:
//
//   4 floats:  power 0.400 mean uncertainty, trigonometric 0.382
//   6 floats:  power 0.287,                  trigonometric 0.275
//   8 floats:  power 0.223,                  trigonometric 0.220
//
// One to five per cent. Not a fix. The control explains why: for a measure made
// of two Dirac spikes, *both* bases are exact to floating point. The extremal
// measures of a truncated moment problem are atomic whatever the basis, so four
// numbers pin down two surfaces perfectly and a continuous slab hardly at all.
// The looseness belongs to the number of moments, not to the basis.
//
// Trigonometric moments do win on conditioning, which is the defensible reading
// of §1: at eight floats the power basis went numerically degenerate on a
// narrow slab where the trigonometric one held up. That matters if the moment
// count ever grows.
//
// # What would actually fix it
//
// Capping the density. The reconstruction is free to answer with Dirac spikes,
// and this content never produces one: absorbance comes in slabs of bounded
// height, because sigma is finite. Adding that single constraint to the same
// four moments collapses the worst error from 0.277 to under 0.006 — better
// than doubling the moment count, by a wide margin, and free in storage. Even a
// cap that is twice too loose halves the error.
//
// The cap is knowable: the density cannot exceed the summed sigma of the
// volumes on screen, and for a single volume that bound is exact. What is not
// cheap is the reconstruction. Under an L-infinity constraint the extremal
// measures stop being atomic and become bang-bang — unions of intervals at
// either zero or the cap — which is Markov-Krein territory rather than a
// Cholesky and a quadratic. That is a research task, not a substitution.
//
// So: do not switch basis expecting an improvement. If the intermediate-depth
// consumer needs better than this, the order of promise is the density cap
// first, more moments second, and the basis a distant third.

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
