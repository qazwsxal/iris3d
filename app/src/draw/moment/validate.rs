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

/// Turns the moment passes on for every 3D camera.
///
/// Temporary. Which views want moment transparency is properly a question for
/// whatever configures a view, and the answer will come from the actor registry
/// once it has settled. Until then the component has to arrive from somewhere,
/// and putting it here keeps the whole backend inside one directory.
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

    /// A tint of 1 on a channel absorbs nothing on it, and a tint of 0 absorbs
    /// at the full rate. That is what makes `tint` a transmission rather than a
    /// surface colour.
    #[test]
    fn tint_is_a_transmission() {
        assert_eq!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 1.0), 1.0);
        assert!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 0.0) < 1.0);
    }
}
