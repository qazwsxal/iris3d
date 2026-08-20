//! Chemistry: facts about the periodic table.
//!
//! An element's radius and its conventional colour are properties of the
//! element, not of a pipeline or of any way of computing with it. Two rendering
//! pathways disagree about *how* to put a sphere on screen and agree completely
//! about how big it is and what colour it should be, and a filter tinting by
//! element wants the same answer a renderer does.
//!
//! So this is a leaf module that depends on nothing in the crate. It sits here
//! rather than under `iris3d_draw` because
//! `iris3d_filter::colormap` reads the same table, and a
//! periodic table under the renderer would make every filter that mentions an
//! element depend on the renderer.
//!
//! Tessellation deliberately does **not** live here. The default backend merges
//! spheres and cylinders into one mesh; one that instanced a single sphere per
//! atom, or raytraced analytic ones, would share no geometry code with it. That
//! is the usual outcome, and the reason actors belong to a backend.

use bevy::prelude::*;

/// Covalent radii by atomic number, in ångströms, for the common elements.
/// Anything unlisted falls back to carbon.
pub fn radius(atomic_number: u32) -> f32 {
    match atomic_number {
        1 => 0.31,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        _ => 0.76,
    }
}

/// Standard CPK colouring.
///
/// Returned **linear**, not sRGB. The stops are quoted in sRGB as they are
/// everywhere else, but every consumer wants linear: vertex colours reach the
/// shader untouched and `pbr_fragment.wgsl` assigns them straight to
/// `base_color`, and a `StandardMaterial`'s `base_color` is converted on the
/// way in. Handing over sRGB values renders everything far too bright.
pub fn colour(atomic_number: u32) -> [f32; 4] {
    let rgb = match atomic_number {
        1 => [0.95, 0.95, 0.95],
        6 => [0.25, 0.25, 0.28],
        7 => [0.19, 0.31, 0.97],
        8 => [0.94, 0.15, 0.10],
        9 => [0.56, 0.88, 0.31],
        15 => [1.00, 0.50, 0.00],
        16 => [0.90, 0.78, 0.19],
        17 => [0.12, 0.94, 0.12],
        _ => [0.85, 0.45, 0.85],
    };
    Color::srgb(rgb[0], rgb[1], rgb[2])
        .to_linear()
        .to_f32_array()
}
