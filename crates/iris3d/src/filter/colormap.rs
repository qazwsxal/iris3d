//! Scalars in, colours out.
//!
//! The first filter, and the smallest one worth having. It takes an array of
//! numbers and produces one linear RGB triple per element, which an actor binds
//! as vertex colours without knowing where they came from.
//!
//! # Why this is a filter and not a setting on an actor
//!
//! A map, a range and a flat colour carried by every actor and applied inside
//! every backend's draw system would put "which ramp" in the same place as "how
//! to rasterise", and leave an actor colourable exactly one way: by one bound
//! scalar, through one built-in ramp.
//!
//! As a filter it composes. Colour by a field the client computed, by a residue
//! index a `cartoon` emitted, by the output of another filter — and the actor is
//! unchanged, because all it ever sees is an RGB array.
//!
//! # Linear, not sRGB
//!
//! Colour maps are *published* in sRGB, and that is the space anyone reading a
//! hex value means. Every consumer wants linear: a vertex colour reaches the
//! shader untouched and `pbr_fragment.wgsl` assigns it straight to `base_color`.
//! So the conversion happens here, once, at the boundary between the two.
//!
//! This was got wrong once, and the symptom is worth recognising: stops handed
//! back unconverted make every ramp brighter and less saturated than the map it
//! names — viridis comes out mid-magenta at the low end and near-white at the
//! top rather than dark purple and yellow.

use bevy::prelude::*;

use crate::scene::DataArray;
use iris3d_data::array::Dtype;
use iris3d_model::{ParamKind, ParamSpec, text, vector};

use super::{
    FilterKind, FilterRegistry, Outcome, OutputKind, OutputSpec, Products, Provenance, Request,
};

/// Which ramp to read a value through.
///
/// Lived on every actor once, as half of a `ColorBy`. It belongs here now: this
/// is the only thing that turns a number into a colour, and a `volume` sampling
/// a ramp texture per step is the one other consumer — which is why [`sample`]
/// and this are shared rather than duplicated.
///
/// `Hash` so a backend can key a cache by map. A pathway that cannot read vertex
/// colours builds a palette of materials and a ramp texture per map instead, and
/// must not rebuild them every frame.
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
    /// named once element colouring routes through [`sample`] like every other
    /// map.
    #[allow(dead_code)]
    ByElement,
}

impl ColorMap {
    /// Inverse of the names in [`MAPS`], for a value arriving from a client.
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

/// The maps offered, in [`ColorMap::from_str`] spelling.
///
/// `element` is deliberately absent. Element colouring is per-atom and comes
/// from the periodic table rather than from a ramp over a range, so it is a
/// different filter rather than an option here — offering it would mean a `map`
/// whose `values` input means something else entirely.
pub(crate) const MAPS: &[&str] = &["viridis", "cool-warm", "grayscale"];

/// What the `colormap` **filter** offers, which is more than [`MAPS`].
///
/// The two lists differ on purpose. `MAPS` is what can be baked into a ramp
/// texture and read at an arbitrary point along it, which is what
/// [`volume`](mod@crate::draw::default::volume) and [`contour`](super::contour) do —
/// they sample per ray step and per vertex against a normalised position in the
/// range.
///
/// The two below are not ramps at all. Both are lookups on a *value*, not on a
/// position between two ends: a chain index of 3 is not "three fifths of the way
/// along" anything, and normalising it would make the colours shift every time
/// the number of chains changed. So they are available where colours are
/// computed per element and absent where a ramp is what is wanted.
const FILTER_MAPS: &[&str] = &[
    "viridis",
    "cool-warm",
    "grayscale",
    "categorical",
    "element",
];

/// A repeating qualitative palette, for colouring by an integer that names a
/// thing rather than measures one — chain, secondary structure, entity.
///
/// Okabe-Ito, which is designed to stay distinguishable under the common forms
/// of colour blindness. Eight entries and it repeats; a structure with more than
/// eight chains reuses colours, which is honest — no eight-colour palette can do
/// otherwise, and the alternative of generating hues on the fly gives neighbours
/// that cannot be told apart.
///
/// Stated in linear RGB, like everything an actor binds. These are the sRGB
/// values converted once here rather than at every use.
const CATEGORICAL: &[[f32; 3]] = &[
    [0.000, 0.180, 0.351], // blue
    [0.902, 0.371, 0.006], // orange
    [0.000, 0.448, 0.288], // bluish green
    [0.871, 0.665, 0.016], // yellow
    [0.021, 0.246, 0.523], // dark blue
    [0.665, 0.155, 0.043], // vermillion
    [0.556, 0.170, 0.400], // reddish purple
    [0.339, 0.610, 0.787], // sky blue
];

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "values",
        label: "values",
        kind: ParamKind::Array {
            // Any numeric type, any shape. A multi-component array reduces to
            // magnitude, exactly as a bound colour array always has.
            dtypes: &[],
            shape: &[],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "map",
        label: "colour map",
        kind: ParamKind::Choice {
            options: FILTER_MAPS,
            default: "viridis",
        },
    },
    // Two components rather than a pair of floats, so "the range" is one thing
    // to set and one thing to read.
    //
    // There is no separate `autoscale` flag. A range whose ends are equal
    // carries no information — every value would land at the same point — so it
    // is the natural spelling of "work it out from the data", and it is the
    // default. That keeps an impossible state out of the map: a flag saying
    // autoscale beside a range that says otherwise.
    ParamSpec {
        id: "range",
        label: "range (equal ends autoscale)",
        kind: ParamKind::Vector {
            components: 2,
            default: &[0.0, 0.0],
            min: -1.0e30,
            max: 1.0e30,
            integral: false,
        },
    },
];

const OUTPUTS: &[OutputSpec] = &[OutputSpec {
    id: "colour",
    label: "colour",
    kind: OutputKind::Array {
        dtype: Some(Dtype::Float32),
        shape: &[0, 3],
    },
    // One colour per value, in order.
    provenance: Provenance::Identity("values"),
}];

pub fn register(registry: &mut FilterRegistry) {
    registry.register(FilterKind {
        id: "colormap",
        label: "colour map",
        params: PARAMS,
        outputs: OUTPUTS,
        run: Some(run),
    });
}

fn run(request: &Request) -> Outcome {
    let mut products = Products::new();
    let Some(values) = request.input("values") else {
        return Outcome::refused("has nothing bound to \"values\"");
    };

    let chosen = text(&request.params, "map", "viridis");
    let range = vector(&request.params, "range", 2);
    let scalars = reduce(values);

    // The two lookups-on-a-value, taken before the ramp path: neither has a
    // range to normalise against. See `FILTER_MAPS`.
    if chosen == "categorical" || chosen == "element" {
        let mut bytes = Vec::with_capacity(scalars.len() * 3 * 4);
        for value in &scalars {
            // Rounded rather than truncated: these arrive as integers widened
            // to f32, and 2.9999998 is a 3 that survived a conversion.
            let key = value.round().max(0.0) as u32;
            let rgb = match chosen {
                "element" => {
                    let rgba = iris3d_data::chem::colour(key);
                    [rgba[0], rgba[1], rgba[2]]
                }
                _ => CATEGORICAL[key as usize % CATEGORICAL.len()],
            };
            for channel in &rgb {
                bytes.extend_from_slice(&channel.to_le_bytes());
            }
        }
        let mut products = Products::new();
        products.insert(
            "colour",
            DataArray::numeric(Dtype::Float32, vec![scalars.len() as u64, 3], bytes).into(),
        );
        return products.into();
    }

    let map = ColorMap::from_str(chosen).unwrap_or_default();

    // Equal ends mean "work it out". A constant field autoscales to a range of
    // zero width, which would divide by zero; it takes the bottom of the map
    // instead, which is at least a colour rather than a NaN.
    let (low, high) = match range[0] < range[1] {
        true => (range[0] as f32, range[1] as f32),
        false => span(&scalars),
    };
    let width = high - low;

    let mut bytes = Vec::with_capacity(scalars.len() * 3 * 4);
    for value in &scalars {
        let t = match width > 0.0 {
            true => (value - low) / width,
            false => 0.0,
        };
        let rgba = sample(map, t);
        for channel in &rgba[..3] {
            bytes.extend_from_slice(&channel.to_le_bytes());
        }
    }

    products.insert(
        "colour",
        DataArray::numeric(Dtype::Float32, vec![scalars.len() as u64, 3], bytes).into(),
    );
    products.into()
}

/// One number per element.
///
/// A multi-component array reduces to magnitude, which is what makes a vector
/// field colourable without saying so anywhere. Magnitude is defensible rather
/// than always right — von Mises is the conventional scalar for a stress tensor
/// — and the answer to that is another filter, not an option here.
fn reduce(array: &DataArray) -> Vec<f32> {
    let values = array.to_f32();
    let components = array.components().max(1) as usize;
    if components == 1 {
        return values;
    }
    values
        .chunks(components)
        .map(|element| element.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// The range the data actually occupies, ignoring anything not finite.
///
/// A NaN in the input would otherwise poison both ends through `min`/`max` and
/// colour the whole array as if it were flat.
fn span(values: &[f32]) -> (f32, f32) {
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for value in values.iter().copied().filter(|v| v.is_finite()) {
        low = low.min(value);
        high = high.max(value);
    }
    match low <= high {
        true => (low, high),
        false => (0.0, 0.0),
    }
}

/// Samples a colour map, returning **linear** RGBA.
///
/// Public because the ramp texture a volume samples has to agree with the
/// vertex colours a mesh gets, so both come from here.
pub(crate) fn sample(map: ColorMap, t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let rgb = match map {
        ColorMap::Viridis => ramp(&VIRIDIS, t),
        ColorMap::CoolWarm => ramp(
            &[
                [0.230, 0.299, 0.754],
                [0.865, 0.865, 0.865],
                [0.706, 0.016, 0.150],
            ],
            t,
        ),
        ColorMap::Grayscale => [t, t, t],
        // Element colouring is per-atom, not a ramp; molecules handle it
        // directly and never reach here.
        ColorMap::ByElement => [0.8, 0.8, 0.85],
    };
    Color::srgb(rgb[0], rgb[1], rgb[2])
        .to_linear()
        .to_f32_array()
}

/// Linearly interpolates between evenly spaced colour stops.
fn ramp(stops: &[[f32; 3]], t: f32) -> [f32; 3] {
    if stops.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let last = stops.len() - 1;
    let scaled = t.clamp(0.0, 1.0) * last as f32;
    let low = scaled.floor() as usize;
    let high = (low + 1).min(last);
    let blend = scaled - low as f32;
    let mut rgb = [0.0; 3];
    for channel in 0..3 {
        rgb[channel] = stops[low][channel] * (1.0 - blend) + stops[high][channel] * blend;
    }
    rgb
}

/// Nine evenly spaced stops, linearly interpolated. Enough to be perceptually
/// honest without carrying a 256-entry table.
///
/// Quoted in **sRGB**, which is how viridis is published and what [`sample`]
/// converts from. Blending between them in that space rather than after the
/// conversion is a small inaccuracy — a strict ramp interpolates in a linear or
/// perceptual space — but with nine stops the two are within a rounding error of
/// each other, and blending as published is the reading that matches the
/// swatches everyone knows.
const VIRIDIS: [[f32; 3]; 9] = [
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.993, 0.906, 0.144],
];

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;
    use iris3d_model::{ParamMap, ParamValue};

    /// Builds a request the way the runner would, so a test exercises the same
    /// entry point the schedule uses.
    fn request(values: Vec<f32>, params: &[(&str, ParamValue)]) -> Request {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let array = DataArray::numeric(Dtype::Float32, vec![values.len() as u64], bytes);

        let mut map = ParamMap::new();
        for (id, value) in params {
            map.insert((*id).to_string(), value.clone());
        }
        let mut inputs = HashMap::new();
        inputs.insert("values", array);
        Request {
            params: map,
            inputs,
        }
    }

    fn produced(products: &Outcome) -> &DataArray {
        products.products["colour"]
            .array()
            .expect("colour is an array")
    }

    fn colours(products: &Outcome) -> Vec<[f32; 3]> {
        produced(products)
            .to_f32()
            .chunks(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect()
    }

    #[test]
    fn produces_one_linear_rgb_triple_per_element() {
        let products = run(&request(vec![0.0, 0.5, 1.0], &[]));
        assert_eq!(produced(&products).shape, vec![3, 3], "[n, 3]");
        assert_eq!(produced(&products).dtype, Dtype::Float32);
        assert_eq!(colours(&products).len(), 3);
    }

    /// The ends of the data land on the ends of the map, which is what
    /// autoscaling means and the only thing a caller can rely on without
    /// pinning a range.
    #[test]
    fn an_unset_range_autoscales_to_the_data() {
        let products = run(&request(vec![10.0, 20.0], &[]));
        let seen = colours(&products);
        assert_eq!(seen[0], sample(ColorMap::Viridis, 0.0)[..3]);
        assert_eq!(seen[1], sample(ColorMap::Viridis, 1.0)[..3]);
    }

    /// A pinned range is what lets two actors share a scale, so it has to win
    /// over what the data happens to contain.
    #[test]
    fn a_pinned_range_is_not_autoscaled() {
        let pinned = ParamValue::Vector(vec![0.0, 100.0]);
        let products = run(&request(vec![0.0, 100.0], &[("range", pinned)]));
        let seen = colours(&products);
        assert_eq!(seen[0], sample(ColorMap::Viridis, 0.0)[..3]);
        assert_eq!(seen[1], sample(ColorMap::Viridis, 1.0)[..3]);

        // The same values under a range twice as wide land halfway, which the
        // autoscaled version could never produce.
        let wide = ParamValue::Vector(vec![0.0, 200.0]);
        let products = run(&request(vec![0.0, 100.0], &[("range", wide)]));
        assert_eq!(colours(&products)[1], sample(ColorMap::Viridis, 0.5)[..3]);
    }

    /// A field that never varies has no range to scale to. Dividing by its
    /// width would produce NaN for every element, which reaches the shader as a
    /// black or absent surface rather than as an error.
    #[test]
    fn a_constant_field_does_not_divide_by_zero() {
        let products = run(&request(vec![7.0, 7.0, 7.0], &[]));
        for rgb in colours(&products) {
            assert!(rgb.iter().all(|c| c.is_finite()), "{rgb:?}");
        }
    }

    /// NaN is ordinary in scientific data — a masked voxel, a residue with no
    /// measurement — and one of them must not decide the range for everything
    /// else.
    #[test]
    fn a_nan_does_not_swallow_the_range() {
        let products = run(&request(vec![0.0, f32::NAN, 1.0], &[]));
        let seen = colours(&products);
        assert_eq!(seen[0], sample(ColorMap::Viridis, 0.0)[..3]);
        assert_eq!(seen[2], sample(ColorMap::Viridis, 1.0)[..3]);
    }

    /// A vector field is colourable with no extra declaration: the reduction is
    /// part of what the filter means.
    #[test]
    fn a_multi_component_array_reduces_to_magnitude() {
        let bytes: Vec<u8> = [3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = DataArray::numeric(Dtype::Float32, vec![2, 3], bytes);
        let mut inputs = HashMap::new();
        inputs.insert("values", array);
        let products = run(&Request {
            params: ParamMap::new(),
            inputs,
        });

        let seen = colours(&products);
        assert_eq!(seen.len(), 2, "one colour per element, not per component");
        // 3-4-0 has magnitude 5, the other is 0, so they take the two ends.
        assert_eq!(seen[1], sample(ColorMap::Viridis, 0.0)[..3]);
        assert_eq!(seen[0], sample(ColorMap::Viridis, 1.0)[..3]);
    }

    /// Colour maps are authored in sRGB and every consumer wants linear, so
    /// [`sample`] has to convert. This went unnoticed for a long time because
    /// the failure is quiet — ramps merely looked washed out — and because
    /// `chem::colour` did convert, so half the renderer was right.
    ///
    /// Checked against the transfer function rather than a recorded number, so
    /// the test says *why* the value is what it is.
    #[test]
    fn colour_maps_are_converted_out_of_srgb() {
        /// The sRGB electro-optical transfer function, from the specification.
        fn to_linear(channel: f32) -> f32 {
            if channel <= 0.04045 {
                channel / 12.92
            } else {
                ((channel + 0.055) / 1.055).powf(2.4)
            }
        }

        // Viridis's darkest stop, which is where the error was most visible:
        // read as linear it displays at roughly twice its intended lightness.
        let low = sample(ColorMap::Viridis, 0.0);
        for (channel, quoted) in VIRIDIS[0].iter().enumerate() {
            let wanted = to_linear(*quoted);
            assert!(
                (low[channel] - wanted).abs() < 1e-4,
                "channel {channel}: got {}, expected {wanted} from sRGB {quoted}",
                low[channel]
            );
        }

        // Converting darkens everything except the endpoints, which are fixed
        // points of the transfer function.
        let mid = sample(ColorMap::Grayscale, 0.5);
        assert!(
            mid[0] < 0.25,
            "mid grey should darken to about 0.21, got {}",
            mid[0]
        );
        assert_eq!(sample(ColorMap::Grayscale, 0.0)[0], 0.0);
        assert!((sample(ColorMap::Grayscale, 1.0)[0] - 1.0).abs() < 1e-6);

        // Alpha is not a colour and must not be run through the curve.
        assert_eq!(low[3], 1.0);
    }

    /// Nothing bound means nothing learned, so the previous contents stand
    /// rather than being replaced by an empty array.
    #[test]
    fn an_unbound_input_produces_nothing() {
        let products = run(&Request {
            params: ParamMap::new(),
            inputs: HashMap::new(),
        });
        assert!(products.is_refusal());
    }
}
