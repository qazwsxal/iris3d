//! The reference image, and the arithmetic behind it.
//!
//! The reference document's §11 is emphatic that step 2 must not be skipped:
//! without a picture you know is right, a moment error and a sign error look
//! the same. For a single convex volume the answer is closed form, so this
//! module holds both the closed form and a scene to compare against it.
//!
//! Two scenes, chosen with `IRIS3D_MOMENT_TEST`:
//!
//! - `1` — a lone neutral sphere and an overlapping tinted pair. Steps 1 and 2.
//! - `nested` — one sphere wholly inside another. Step 4.
//!
//! Both are off unless asked for, so nothing here reaches an ordinary session.
//! `IRIS3D_MOMENT_SIGMA=0` renders the same frame with nothing absorbing, which
//! divides out the background and leaves the transmittance on its own.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::render::view::Msaa;
use bevy::prelude::*;

use super::{MomentTransparency, MomentVolume};

/// Transmittance through a sphere along a ray that misses its centre by
/// `offset`, for one colour channel.
///
/// The chord of a sphere at distance `d` from the centre is `2*sqrt(r^2 - d^2)`,
/// and uniform absorbance over a path of that length transmits
/// `exp(-sigma_channel * chord)`. Rays that miss entirely transmit everything.
///
/// This is what the renderer has to reproduce, to within float error. It is not
/// an approximation of the moment method — at `k = 0` and one convex volume it
/// *is* the moment method, evaluated on the CPU.
// Nothing calls this outside the tests yet. It stays because it is the
// specification: when higher moments arrive they are checked against this, and
// a readback comparison against a rendered frame is the obvious next test.
#[allow(dead_code)]
pub fn sphere_transmittance(radius: f32, offset: f32, sigma: f32, tint: f32) -> f32 {
    if offset.abs() >= radius {
        return 1.0;
    }
    let chord = 2.0 * (radius * radius - offset * offset).sqrt();
    (-sigma * chord * (1.0 - tint)).exp()
}

/// `reconstruct.wgsl`'s `absorbed_fraction`, in Rust, so it can be tested.
///
/// A second implementation of the same arithmetic is a real cost — the two can
/// drift — but the alternative was no test at all. The WGSL runs only on a GPU,
/// inside a pass whose one caller currently asks it a question with a known
/// answer, so a mistake in the Cholesky or the root finding would not have
/// shown up in any picture. Keep the two in step by hand; the tests below are
/// what tell you the algorithm itself is right.
#[allow(dead_code)]
pub fn absorbed_fraction(raw: [f32; 4], at: f32) -> f32 {
    const BIAS: f32 = 3.0e-5;
    let b: Vec<f32> = raw.iter().map(|m| m * (1.0 - BIAS) + 0.5 * BIAS).collect();
    let degenerate = if at > b[0] { 1.0 } else { 0.0 };

    let (l21, l31) = (b[0], b[1]);
    let d2 = b[1] - b[0] * b[0];
    if d2 <= 0.0 {
        return degenerate;
    }
    let l32 = (b[2] - b[0] * b[1]) / d2;
    let d3 = b[3] - l31 * l31 - l32 * l32 * d2;
    if d3 <= 0.0 {
        return degenerate;
    }

    let y2 = at - l21;
    let y3 = at * at - l31 - l32 * y2;
    let c3 = y3 / d3;
    let c2 = y2 / d2 - l32 * c3;
    let c1 = 1.0 - l21 * c2 - l31 * c3;

    let discriminant = c2 * c2 - 4.0 * c3 * c1;
    if c3 == 0.0 || discriminant < 0.0 {
        return degenerate;
    }
    let q = -0.5 * (c2 + c2.signum() * discriminant.sqrt());
    let (first, second) = ((q / c3).min(c1 / q), (q / c3).max(c1 / q));

    if at <= first {
        return 0.0;
    }
    if at <= second {
        return (at * second - b[0] * (at + second) + b[1]) / ((second - first) * (at - first));
    }
    1.0 - (first * second - b[0] * (first + second) + b[1]) / ((at - first) * (at - second))
}

/// Normalised power moments of a uniform measure on `[low, high]`.
///
/// The measure an absorbing slab actually produces: constant density between
/// the two faces, nothing outside them.
#[allow(dead_code)]
pub fn slab_moments(low: f32, high: f32) -> [f32; 4] {
    let mut moments = [0.0; 4];
    for (index, moment) in moments.iter_mut().enumerate() {
        let k = index as i32 + 1;
        *moment = (high.powi(k + 1) - low.powi(k + 1)) / ((k + 1) as f32 * (high - low));
    }
    moments
}

/// Turns the moment passes on for every 3D camera.
///
/// Temporary, and living in the test module rather than beside the plugin
/// because it is not a small version of the real thing. Selecting the moment
/// pathway is a startup decision for the whole app — see
/// [`MomentTransparency`] — so the real wiring adds this backend's plugins or
/// does not, and never asks per camera. This exists only so the render half can
/// be run and measured before that decision has anywhere to live.
pub fn enable_on_cameras(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<MomentTransparency>)>,
) {
    for camera in &cameras {
        // MSAA off with it. The moment target would have to be multisampled to
        // match the depth buffer it clamps against, and the resolve would have
        // to read per sample; neither is built. `prepare_moment_textures` skips
        // a multisampled view rather than rendering something wrong, so without
        // this the volumes would simply not appear.
        commands.entity(camera).insert((MomentTransparency, Msaa::Off));

        // Tonemapping runs after the resolve, so it would bend the recorded
        // pixel values away from the transmittance that produced them. Turning
        // it off makes the frame a linear read-out, which is what lets a
        // capture be divided by a sigma = 0 capture and checked against
        // `sphere_transmittance`. Test scenes only — this is not a rendering
        // opinion.
        if in_test_scene() {
            commands.entity(camera).insert(Tonemapping::None);
        }
    }
}

fn in_test_scene() -> bool {
    scene_name().is_some()
}

/// Which test scene was asked for, if any.
///
/// `1` is the original trio and `nested` is the step 4 scene. Kept as separate
/// scenes rather than one crowded one because the camera frames whatever is
/// there: adding geometry to a scene moves everything already in it, and the
/// pixel checks would have to be recalibrated every time.
fn scene_name() -> Option<String> {
    std::env::var("IRIS3D_MOMENT_TEST").ok().filter(|name| !name.is_empty())
}

/// Says so when a view asked for moment transparency and cannot have it.
///
/// The render world quietly skips such a view, which on its own would look like
/// the backend being broken.
pub fn warn_about_msaa(cameras: Query<&Msaa, (With<MomentTransparency>, Changed<Msaa>)>) {
    for msaa in &cameras {
        if msaa.samples() > 1 {
            warn!(
                "draw: moment transparency does not support MSAA yet ({}x requested); \
                 absorbing volumes will not be drawn on this view",
                msaa.samples()
            );
        }
    }
}

/// Radius of the test sphere. Also the number the analytic check uses, so the
/// two cannot drift apart.
pub const TEST_RADIUS: f32 = 1.0;
pub const TEST_SIGMA: f32 = 1.5;

/// Three spheres: one alone, and two overlapping.
///
/// The lone sphere is the analytic reference of §11 step 2. The overlapping
/// pair is the part no amount of triangle sorting gets right, and it must look
/// identical whichever order the two are drawn in — which, since nothing here
/// sorts, it does by construction. Their intersection must read as the sum of
/// the two absorbances, so it is darker than either.
pub fn spawn_test_scene(commands: Commands, meshes: ResMut<Assets<Mesh>>) {
    match scene_name().as_deref() {
        Some("1") => spawn_trio(commands, meshes),
        Some("nested") => spawn_nested(commands, meshes),
        Some(other) => warn!("draw: no moment test scene called \"{other}\""),
        None => {}
    }
}

/// Radius of the inner shell in the nested scene, as a fraction of
/// [`TEST_RADIUS`].
pub const NESTED_INNER: f32 = 0.6;

/// One closed mesh wholly inside another — step 4 of the build order.
///
/// A ray through the middle now crosses four faces rather than two, in the
/// order front, front, back, back. Nothing pairs them up: each deposits
/// `±sigma * z` and the additive blend is left to cancel what should cancel.
/// If the sign rule is right this needs no code at all, which is the claim
/// being tested.
///
/// The answer stays closed form. Two concentric spheres absorb the sum of two
/// chords, so the picture must show a distinct step at the inner silhouette —
/// and that step is what a broken pairing would smear or invert.
fn spawn_nested(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let sigma = configured_sigma();
    let outer = meshes.add(Sphere::new(TEST_RADIUS).mesh().ico(5).unwrap());
    let inner = meshes.add(Sphere::new(TEST_RADIUS * NESTED_INNER).mesh().ico(5).unwrap());

    for mesh in [outer, inner] {
        commands.spawn((
            Mesh3d(mesh),
            Transform::IDENTITY,
            MomentVolume {
                sigma,
                tint: Vec3::ZERO,
            },
        ));
    }

    info!("draw: moment nested test scene spawned");
}

fn configured_sigma() -> f32 {
    std::env::var("IRIS3D_MOMENT_SIGMA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(TEST_SIGMA)
}

fn spawn_trio(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {

    // Overridable so the same frame can be rendered at sigma = 0, which
    // transmits everything and is therefore the background on its own. Dividing
    // one render by the other isolates `T` per pixel, which is what the
    // analytic check needs — the camera frames the scene bounds, so the volumes
    // have to stay put for the two frames to line up.
    let sigma = configured_sigma();

    // Deliberately no material. A `Mesh3d` with no `MeshMaterial3d` is never
    // queued into the opaque or transparent phases, so the only thing that
    // draws these is the moment pass.
    let sphere = meshes.add(Sphere::new(TEST_RADIUS).mesh().ico(5).unwrap());

    commands.spawn((
        Mesh3d(sphere.clone()),
        Transform::from_xyz(-2.5, 0.0, 0.0),
        MomentVolume {
            sigma,
            // Neutral, so the picture is a direct read-out of thickness and can
            // be compared against `sphere_transmittance` by eye.
            tint: Vec3::ZERO,
        },
    ));

    commands.spawn((
        Mesh3d(sphere.clone()),
        Transform::from_xyz(0.6, 0.0, 0.0),
        MomentVolume {
            sigma,
            tint: Vec3::new(0.9, 0.15, 0.15),
        },
    ));
    commands.spawn((
        Mesh3d(sphere),
        Transform::from_xyz(1.9, 0.0, 0.0),
        MomentVolume {
            sigma,
            tint: Vec3::new(0.15, 0.35, 0.95),
        },
    ));

    info!("draw: moment test scene spawned (IRIS3D_MOMENT_TEST=1)");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray straight through the middle travels a full diameter.
    #[test]
    fn the_centre_is_a_full_diameter() {
        let expected = (-TEST_SIGMA * 2.0 * TEST_RADIUS).exp();
        let got = sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 0.0);
        assert!((got - expected).abs() < 1e-6, "got {got}, want {expected}");
    }

    /// The silhouette is where the chord vanishes, so the volume must be
    /// completely clear there. This is the edge a sign error shows up at: get
    /// the winding backwards and the rim goes black instead.
    #[test]
    fn the_rim_is_clear() {
        assert_eq!(sphere_transmittance(TEST_RADIUS, TEST_RADIUS, TEST_SIGMA, 0.0), 1.0);
        assert_eq!(sphere_transmittance(TEST_RADIUS, 2.0, TEST_SIGMA, 0.0), 1.0);
    }

    /// Thickness falls off towards the rim, so transmittance must rise
    /// monotonically with the offset. A picture that does the opposite has the
    /// front and back contributions swapped.
    #[test]
    fn transmittance_rises_towards_the_rim() {
        let mut previous = 0.0;
        for step in 0..=10 {
            let offset = step as f32 / 10.0 * TEST_RADIUS;
            let value = sphere_transmittance(TEST_RADIUS, offset, TEST_SIGMA, 0.0);
            assert!(value > previous, "at offset {offset}: {value} <= {previous}");
            previous = value;
        }
    }

    /// Absorbance is additive, so two overlapping volumes transmit the product
    /// of what each transmits alone. This is the property the whole method
    /// rests on, and it is order-independent because both addition and
    /// multiplication commute.
    ///
    /// It is also the exact path the shaders take: they accumulate `A` with an
    /// additive blend and then apply `exp(-A)` once, rather than transmitting
    /// twice.
    #[test]
    fn overlapping_volumes_multiply() {
        let first = sphere_transmittance(TEST_RADIUS, 0.2, TEST_SIGMA, 0.0);
        let second = sphere_transmittance(TEST_RADIUS, 0.5, 1.0, 0.0);

        // What the moment target holds after both have been drawn.
        let accumulated = -first.ln() + -second.ln();
        // What the resolve pass makes of it.
        let resolved = (-accumulated).exp();

        assert!(
            (resolved - first * second).abs() < 1e-6,
            "accumulating then resolving gave {resolved}, transmitting twice gives {}",
            first * second
        );
    }

    /// Two concentric shells transmit the product of what each transmits, which
    /// is the same additivity as the overlapping case — nesting is not a
    /// special geometry, only a special arrangement.
    #[test]
    fn nested_shells_compose() {
        let inner_radius = TEST_RADIUS * NESTED_INNER;
        for offset in [0.0, 0.3, 0.55] {
            let expected = sphere_transmittance(TEST_RADIUS, offset, TEST_SIGMA, 0.0)
                * sphere_transmittance(inner_radius, offset, TEST_SIGMA, 0.0);
            let chords = 2.0 * (TEST_RADIUS.powi(2) - offset * offset).sqrt()
                + 2.0 * (inner_radius.powi(2) - offset * offset).sqrt();
            let got = (-TEST_SIGMA * chords).exp();
            assert!((got - expected).abs() < 1e-6, "at {offset}: {got} vs {expected}");
        }
    }

    /// The inner silhouette is a step, not a smooth join: crossing it adds a
    /// whole second pair of faces. A pairing bug shows up here first, because
    /// this is where the number of crossings changes.
    #[test]
    fn the_inner_silhouette_is_a_step() {
        let inner_radius = TEST_RADIUS * NESTED_INNER;
        let just_inside = |d: f32| {
            sphere_transmittance(TEST_RADIUS, d, TEST_SIGMA, 0.0)
                * sphere_transmittance(inner_radius, d, TEST_SIGMA, 0.0)
        };
        let inside = just_inside(inner_radius - 0.02);
        let outside = just_inside(inner_radius + 0.02);
        assert!(
            outside > inside * 1.2,
            "expected a visible step at the inner rim, got {inside} then {outside}"
        );
    }

    /// Nothing lies in front of the near face, and everything lies in front of
    /// the far one. These are the two ends the reconstruction has to pin down
    /// before anything between them is worth discussing.
    #[test]
    fn the_ends_of_a_slab_are_recovered() {
        let moments = slab_moments(0.3, 0.8);
        assert_eq!(absorbed_fraction(moments, 0.3), 0.0, "at the near face");
        assert!(
            absorbed_fraction(moments, 0.3 - 0.05) == 0.0,
            "in front of the slab"
        );
        assert!(
            absorbed_fraction(moments, 1.0) > 0.98,
            "past the far face: got {}",
            absorbed_fraction(moments, 1.0)
        );
    }

    /// The reconstruction never claims more absorbance than there is. It is a
    /// lower bound, so every error is towards a volume that reads too bright,
    /// never too dark.
    #[test]
    fn the_reconstruction_never_overestimates() {
        for (low, high) in [(0.4, 0.6), (0.3, 0.8), (0.05, 0.95), (0.45, 0.55)] {
            let moments = slab_moments(low, high);
            for step in 0..=20 {
                let at = step as f32 / 20.0;
                let truth = ((at - low) / (high - low)).clamp(0.0, 1.0);
                let got = absorbed_fraction(moments, at);
                assert!(
                    got <= truth + 1e-3,
                    "[{low}, {high}] at {at}: {got} exceeds the true {truth}"
                );
            }
        }
    }

    /// How badly four power moments bound a *continuous* measure, recorded
    /// rather than papered over.
    ///
    /// At its own midpoint a uniform slab should reconstruct at one half. It
    /// does not come close, because the bound is exact only for measures living
    /// on two points and a slab is the opposite of that. This is worth pinning
    /// down: the reference document's §3.3 claims uniform interior absorbance
    /// is *easier* to reconstruct than surfaces, and for the bound that is
    /// backwards. It is easier to accumulate.
    ///
    /// The test asserts the error is real and bounded rather than asserting it
    /// is small. If a later change improves it, this fails and should be
    /// tightened.
    #[test]
    fn a_continuous_slab_is_bounded_loosely() {
        let moments = slab_moments(0.4, 0.6);
        let midpoint = absorbed_fraction(moments, 0.5);
        assert!(
            (0.2..0.35).contains(&midpoint),
            "expected the known loose bound near 0.26, got {midpoint}"
        );
    }

    /// The looseness at the far end is what made the fullscreen resolve use the
    /// total instead of asking the moments. A volume spanning most of the
    /// domain loses several per cent of its absorbance there, and since the
    /// resolve's query point is always the far end, that was pure error.
    #[test]
    fn the_far_end_is_where_the_bound_costs_most() {
        let narrow = absorbed_fraction(slab_moments(0.45, 0.55), 1.0);
        let wide = absorbed_fraction(slab_moments(0.05, 0.95), 1.0);
        assert!(narrow > 0.999, "a thin slab is bounded tightly: {narrow}");
        assert!(
            (0.9..0.95).contains(&wide),
            "a wide one is not: {wide}"
        );
    }

    /// A tint of 1 on a channel absorbs nothing on it, and a tint of 0 absorbs
    /// at the full rate. That is what makes `tint` a transmission rather than a
    /// surface colour.
    #[test]
    fn tint_is_a_transmission() {
        assert_eq!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 1.0), 1.0);
        assert!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 0.0) < 1.0);
    }
}
