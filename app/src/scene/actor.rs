//! How an object gets drawn.
//!
//! Actors are **entities of their own**, linked to the object they draw by
//! [`ActorOf`](super::ActorOf), so one dataset can carry several at once — a
//! protein as cartoon *and* licorice, a grid as outline *and* volume. That
//! also makes it possible to run two rendering approaches over the same data
//! side by side and compare them, which is a goal rather than an accident.
//!
//! What kinds of actor exist is not decided here: the running backend declares
//! them, and [`registry`](super::registry) holds the declarations. This module
//! keeps only what every kind has regardless of pathway, which today is
//! colouring.
//!
//! Nothing here draws anything. A kind is a pair of systems inside a backend:
//! one queries its own style component alongside the actor's bindings and
//! builds whatever that pathway draws, the other says what a change to its
//! parameters invalidated. How the same arrays are best mapped onto GPU
//! primitives depends on the pipeline, which is why kinds belong to a backend
//! rather than sitting above them all. See [`crate::draw`].

use bevy::prelude::*;

/// How an actor takes its colour.
///
/// *What* it is coloured by is not here: that is the array bound to the kind's
/// colour input. This is only the presentation — which ramp, over what range,
/// and what to paint when nothing is bound. It used to carry a `field` naming a
/// field on the object being drawn, which stopped meaning anything once the
/// kinds read bound arrays instead.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct ColorBy {
    pub map: ColorMap,
    /// Value range mapped across the colour map. `None` autoscales to the bound
    /// array's own range.
    pub range: Option<(f32, f32)>,
    /// Used when no colour array is bound.
    pub flat: Color,
}

impl Default for ColorBy {
    fn default() -> Self {
        Self {
            map: ColorMap::Viridis,
            range: None,
            flat: Color::srgb(0.8, 0.8, 0.85),
        }
    }
}

// `Hash` so a backend can key a cache by map: the solari pathway builds a
// palette of materials and a ramp texture per map and must not rebuild them
// every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorMap {
    /// Perceptually uniform; a safe default for scalar fields.
    #[default]
    Viridis,
    /// Diverging, for signed quantities about zero.
    CoolWarm,
    Grayscale,
    /// Standard element colouring for molecular data.
    ///
    /// Never selected today: molecules apply CPK colours directly and never
    /// consult the map, so this variant is what that behaviour *should* be
    /// named once element colouring routes through `draw::sample` like every
    /// other map.
    #[allow(dead_code)]
    ByElement,
}

impl ColorMap {
    pub fn as_str(self) -> &'static str {
        match self {
            ColorMap::Viridis => "viridis",
            ColorMap::CoolWarm => "cool-warm",
            ColorMap::Grayscale => "grayscale",
            ColorMap::ByElement => "element",
        }
    }

    /// Inverse of [`as_str`](Self::as_str), for names arriving from a client.
    // Nothing sends a colour map over the wire yet.
    #[allow(dead_code)]
    pub fn from_str(name: &str) -> Option<Self> {
        Some(match name {
            "viridis" => ColorMap::Viridis,
            "cool-warm" => ColorMap::CoolWarm,
            "grayscale" => ColorMap::Grayscale,
            "element" => ColorMap::ByElement,
            _ => return None,
        })
    }
}
