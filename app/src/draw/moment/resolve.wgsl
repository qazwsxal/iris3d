// Resolve pass: moments back to transmittance.
//
// The moments describe *where along the ray* the absorbance sits, so this can
// ask for the absorbance in front of any depth rather than only the total. It
// asks at the opaque surface, which is where the pixel behind it actually is.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

struct Bounds {
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var moment_texture: texture_2d<f32>;
@group(0) @binding(2) var totals_texture: texture_2d<f32>;
@group(0) @binding(3) var opaque_depth: texture_depth_2d;
@group(0) @binding(4) var<uniform> bounds: Bounds;

// Interpolation weight towards the fixed vector below.
//
// Without it the Hankel matrix loses positive definiteness and the Cholesky
// decomposition fails, which shows on screen as elongated bands of broken
// pixels rather than as noise. Peters and Klein use 3e-5 for moments quantised
// to 16 bits; these are full f32, so this is more headroom than needed — but
// the reference document's §5 warns that the published value assumes low
// overdraw, and raising it only costs a little accuracy.
const BIAS: f32 = 3.0e-5;

// The moment vector biased towards: half the measure at each end of the
// domain, whose normalised power moments are all one half.
const BIAS_TARGET: vec4<f32> = vec4(0.5, 0.5, 0.5, 0.5);

// The fraction of the absorbance lying in front of `at`.
//
// Algorithm 3 ("Hamburger 4MSM") of Peters and Klein, Moment Shadow Mapping,
// I3D 2015 — the same reconstruction MBOIT uses for power moments. Returns a
// lower bound rather than an estimate, so volumes read slightly too bright
// rather than too dark, which is the error direction §5 predicts.
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

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Loaded by integer coordinate, never sampled. Filtering would average the
    // absorbance of neighbouring pixels, whose rays pass through different
    // parts of the volume.
    let coord = vec2<i32>(floor(in.position.xy));
    let raw = textureLoad(moment_texture, coord, 0);
    let totals = textureLoad(totals_texture, coord, 0);

    // A negative total is not physical. It means the front and back faces did
    // not pair up — an open mesh, or geometry clipped by the near plane while
    // the camera sits inside the volume. Clamping keeps the failure to a
    // too-clear volume rather than an `exp` of a large positive number, which
    // would bloom to white.
    let b0 = totals.a;
    if b0 <= 1e-6 {
        return vec4(1.0);
    }

    // The opaque surface is where the light being absorbed actually comes from,
    // so that is the depth to reconstruct at. The accumulation already clamped
    // every interval to it, so this comes out at essentially 1 for a volume
    // wholly in front of the surface — which is what makes this agree with the
    // k = 0 pass wherever k = 0 was already right.
    let depth = textureLoad(opaque_depth, coord, 0);
    let span = max(bounds.far - bounds.near, 1e-6);
    var at = 1.0;
    if depth > 0.0 {
        // Same unprojection as the accumulation pass, via the fullscreen
        // triangle's own clip position.
        let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
        let unprojected = view.view_from_clip * vec4(ndc, depth, 1.0);
        let opaque_z = -unprojected.z / unprojected.w;
        at = clamp((opaque_z - bounds.near) / span, 0.0, 1.0);
    }

    let fraction = clamp(absorbed_fraction(raw / b0, at), 0.0, 1.0);
    let absorbance = max(totals.rgb, vec3(0.0)) * fraction;

    // The pipeline blends this multiplicatively: dst = dst * T. Pure absorption
    // dims what is behind and adds nothing of its own.
    return vec4(exp(-absorbance), 1.0);
}
