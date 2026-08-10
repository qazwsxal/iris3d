//! The reference image, and the arithmetic behind it.
//!
//! The reference document's §11 is emphatic that step 2 must not be skipped:
//! without a picture you know is right, a moment error and a sign error look
//! the same. For a single convex volume the answer is closed form, so this
//! module holds both the closed form and a scene to compare against it.
//!
//! Set `IRIS3D_MOMENT_TEST=1` to get the scene. It is off otherwise, so nothing
//! here reaches an ordinary session.

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
    std::env::var("IRIS3D_MOMENT_TEST").as_deref() == Ok("1")
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
pub fn spawn_test_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    if std::env::var("IRIS3D_MOMENT_TEST").as_deref() != Ok("1") {
        return;
    }

    // Overridable so the same frame can be rendered at sigma = 0, which
    // transmits everything and is therefore the background on its own. Dividing
    // one render by the other isolates `T` per pixel, which is what the
    // analytic check needs — the camera frames the scene bounds, so the volumes
    // have to stay put for the two frames to line up.
    let sigma = std::env::var("IRIS3D_MOMENT_SIGMA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(TEST_SIGMA);

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

    /// A tint of 1 on a channel absorbs nothing on it, and a tint of 0 absorbs
    /// at the full rate. That is what makes `tint` a transmission rather than a
    /// surface colour.
    #[test]
    fn tint_is_a_transmission() {
        assert_eq!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 1.0), 1.0);
        assert!(sphere_transmittance(TEST_RADIUS, 0.0, TEST_SIGMA, 0.0) < 1.0);
    }
}
