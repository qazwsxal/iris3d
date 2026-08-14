//! Rendering backends.
//!
//! A **backend** is a whole rendering pathway: one pipeline, together with the
//! actor kinds built for it. Backends are mutually exclusive, and which one
//! runs is decided once at launch — not per camera, not per object, not per
//! actor. Two techniques that composite differently cannot share a frame
//! correctly, so choosing once removes the whole class of interop questions
//! rather than answering them one at a time.
//!
//! This module is the part every backend shares, and it draws nothing itself:
//! what makes an actor out of date, how a bound handle resolves to an array,
//! how values become colours, and what order the per-frame work runs in. Each
//! backend lives in its own directory below, owns whatever GPU data it
//! produces, and registers its own actor kinds. How a dataset is best mapped
//! onto GPU primitives depends on the pipeline, which is why the actors belong
//! to the backend rather than sitting above it.
//!
//! What each kind needs is declared, not assumed: a kind states its inputs as
//! array parameters, so a client is told that `points` wants `float32 [n, 3]`
//! rather than having to name an array "positions" and hope.
//!
//! # One pathway
//!
//! [`default`] is the only one, and it accumulates moments: opaque geometry
//! goes through Bevy's ordinary passes, and anything transmitting deposits
//! absorbance into a shared buffer instead of blending, so a structure inside
//! the density map it was built from composes correctly with nothing sorted.
//!
//! Two others have been and gone. A plain `Mesh3d`-per-actor baseline on the
//! standard pipeline could not compose a mesh with a volume correctly, which is
//! the thing this project is actually for. A `bevy_solari` raytracing pathway
//! had no transparency and no volumes, and having to keep every kind working
//! under both was shaping the design of things that had nothing to do with
//! raytracing. Both are recoverable from history.
//!
//! The **seam stays** even with one pathway behind it: this module is the shared
//! layer and [`default`] is a pathway directory. Which technique iris3d should
//! settle on is still an open question, and flattening the two together would
//! answer it by accident. What has gone is only the machinery for reconciling
//! *two at once* — the `Backend` enum, the `--backend` flag, and the `shared`
//! flag on a kind.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::WgpuFeatures;

use crate::scene::actor::ColorMap;

use crate::scene::registry::{ActorKindId, ActorRegistry, Bindings};
use crate::scene::{ColorBy, DataArray, DataStore, Subset};

mod atoms;
mod cartoon;
mod default;
mod elements;
mod glycan;
mod probe;
#[cfg(test)]
mod smoke;

/// What to call the running pathway in a message. Also what the registry
/// reports as the backend a kind came from.
///
/// A constant rather than a method on an enum of one. It stays a *name* rather
/// than becoming implicit, because the registry reports it to clients and
/// "which pathway refused" is a question that outlives any particular pathway.
pub const BACKEND: &str = "default";

/// GPU features without which the running pathway cannot draw at all.
///
/// Asked once at startup and answered by refusing, never by degrading — see
/// [`probe`], which enforces that.
///
/// Additive blending into a 32-bit float target is what the whole method rests
/// on, and it is not in the WebGPU baseline. fp32 is not negotiable here: a
/// moment is a difference of two O(1) values, so a thin shell cancels
/// catastrophically in fp16 — see `ref/mboit-bevy-reference.md` §8. Without the
/// feature there is no degraded version to fall back to, only a wrong one.
pub const REQUIRES: WgpuFeatures = WgpuFeatures::FLOAT32_BLENDABLE;

/// What about an actor's drawable output is out of date.
///
/// Graded rather than a single flag, because the three differ by orders of
/// magnitude in cost. Re-tessellating a protein to drag a colour-map slider
/// meant rebuilding a merged mesh of tens of thousands of vertices per frame to
/// change four bytes each.
///
/// Flags accumulate and are cleared together once the backend has had its
/// turn. Geometry subsumes colour: a rebuild produces vertex colours anyway.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Dirty {
    /// Vertex buffers must be rebuilt from the data.
    pub geometry: bool,
    /// Vertex colours must be recomputed. The geometry itself stands, so the
    /// vertex count is unchanged and colours can be written in place.
    pub colour: bool,
    /// A material property changed and nothing about the mesh did.
    pub material: bool,
}

impl Dirty {
    /// Everything, for an actor that has never been drawn.
    pub const ALL: Self = Self {
        geometry: true,
        colour: true,
        material: true,
    };
    pub const GEOMETRY: Self = Self {
        geometry: true,
        ..Self::NOTHING
    };
    pub const COLOUR: Self = Self {
        colour: true,
        ..Self::NOTHING
    };
    pub const MATERIAL: Self = Self {
        material: true,
        ..Self::NOTHING
    };
    const NOTHING: Self = Self {
        geometry: false,
        colour: false,
        material: false,
    };

    pub fn any(self) -> bool {
        self.geometry || self.colour || self.material
    }
}

/// Records that part of an actor needs redoing.
///
/// Merges rather than overwrites: several systems mark independently in one
/// tick — the generic classifier and each actor kind's own — and an `insert`
/// would let whichever ran last drop the others' findings. `or_default` also
/// means no kind has to arrange for the component to exist first.
pub(crate) fn mark(commands: &mut Commands, entity: Entity, what: Dirty) {
    commands
        .entity(entity)
        .entry::<Dirty>()
        .or_default()
        .and_modify(move |mut dirty| {
            dirty.geometry |= what.geometry;
            dirty.colour |= what.colour;
            dirty.material |= what.material;
        });
}

/// What every actor has, whichever backend is running: its own style, how it is
/// coloured, how much of the data it draws, what it draws, and what is out of
/// date.
///
/// A backend extends this with whatever *it* produced last time, which is what
/// makes reuse rather than reallocation possible — and is precisely the part
/// that depends on the pipeline. Each kind declares its own; see the `Drawable` alias in any of them.
pub(crate) type Actor<'a, Style> = (
    Entity,
    &'a Style,
    &'a ColorBy,
    &'a Subset,
    &'a Bindings,
    &'a Dirty,
);

/// Resolves one of an actor's bound inputs to the array behind it.
///
/// Two hops, because a handle is what a client names and an `AssetId` is what
/// the renderer needs: the store maps the first to the second. `None` covers
/// both an input nobody bound — normal for an optional one — and a handle whose
/// array has since been released.
pub(crate) fn bound<'a>(
    bindings: &Bindings,
    input: &str,
    store: &DataStore,
    arrays: &'a Assets<DataArray>,
) -> Option<&'a DataArray> {
    arrays.get(&store.get(bindings.get(input)?)?.handle)
}

/// Ordering label for the systems that decide what is out of date, so every one
/// of them has marked before anything reads the result.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Invalidate;

/// Ordering label for the systems that build an actor's drawable output.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Draw;

/// Ordering label for giving each placement whatever the actor it copies now
/// holds. After [`Draw`], so a placement picks up a handle on the same frame
/// the actor gets one rather than a frame late.
///
/// Not called `Copy`: a type of that name would shadow the trait for the whole
/// module, and `Dirty` derives it.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Place;

/// Runs the backend, plus the work that sits above whichever one it is.
pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        // Declaring the kinds is what makes them exist at all — `scene` holds
        // no list of its own. `init_resource` rather than `insert_resource`
        // so the backend plugin can register into it whichever order the two
        // are added in.
        app.init_resource::<ActorRegistry>();
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .served_by(BACKEND);

        // The order lives in sets rather than one `.chain()`, because the
        // systems being ordered are added by two different plugins and neither
        // can name the other's.
        app.configure_sets(
            Update,
            (
                // Actors are spawned by the scene during Update, and their
                // style components are derived from the parameters, so an actor
                // has no style at all until this has run.
                Invalidate.after(crate::scene::registry::apply_actor_params),
                Draw.after(Invalidate),
                Place.after(Draw),
            ),
        )
        .add_systems(
            Update,
            (mark_dirty.in_set(Invalidate), clear_dirty.after(Place)),
        );

        // Exactly one. Backends are pathways, not layers, so adding a second
        // here would be a mistake rather than a feature.
        app.add_plugins(default::MomentBackendPlugin);
    }

    /// Runs after every plugin has built, which is the first moment the render
    /// app holds the adapter this pathway will actually draw with.
    fn finish(&self, app: &mut App) {
        probe::refuse_unsupported(app);
    }
}

/// Flags what needs redoing, for the reasons any backend would agree on.
///
/// Style parameters are not among them — what a parameter affects is the actor
/// kind's business, so each classifies its own. See the `invalidate` system in
/// each of them.
#[allow(clippy::too_many_arguments)]
fn mark_dirty(
    mut commands: Commands,
    new_actors: Query<Entity, Added<ActorKindId>>,
    recoloured: Query<Entity, (With<ActorKindId>, Changed<ColorBy>)>,
    resubset: Query<Entity, (With<ActorKindId>, Changed<Subset>)>,
    rebound: Query<Entity, (With<ActorKindId>, Changed<Bindings>)>,
    mut array_events: MessageReader<AssetEvent<DataArray>>,
    store: Res<DataStore>,
    bindings: Query<(Entity, &Bindings)>,
) {
    for entity in &new_actors {
        mark(&mut commands, entity, Dirty::ALL);
    }

    // Colour only: the vertices stay exactly where they are, so the mesh is
    // repainted rather than rebuilt. For a merged protein that is the
    // difference between writing a colour per vertex and re-tessellating every
    // atom and bond.
    for entity in &recoloured {
        mark(&mut commands, entity, Dirty::COLOUR);
    }

    // A different selection means different vertices, so this is a rebuild
    // rather than a repaint.
    for entity in &resubset {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }

    // Binding a different array is new data, not a new setting: the vertex count
    // itself changes, so there is nothing to write in place.
    for entity in &rebound {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }

    // Array contents can be rewritten without any binding changing, so watch the
    // assets directly.
    let modified: Vec<_> = array_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    if modified.is_empty() {
        return;
    }
    // Asking each actor what it binds, rather than each object what it holds.
    // That is what keeps this right for shared data: one array feeding three
    // actors redraws all three, wherever in the tree they sit.
    for (actor, bound) in &bindings {
        let touched = bound.0.values().any(|id| {
            store
                .get(*id)
                .is_some_and(|held| modified.contains(&held.handle.id()))
        });
        if touched {
            mark(&mut commands, actor, Dirty::GEOMETRY);
        }
    }
}

/// Clears the flags once the backend has had a chance at them.
///
/// Done here rather than in each actor kind because an actor is only handled by
/// the one kind that understands it, and the others must not clear flags they
/// ignored. Cleared in place rather than removed, so the component stays put and
/// marking never costs an archetype move.
fn clear_dirty(mut dirty: Query<&mut Dirty>) {
    for mut dirty in &mut dirty {
        if dirty.any() {
            *dirty = Dirty::default();
        }
    }
}

/// Maps a bound array onto vertex colours.
///
/// No `Field` and no name lookup: a bound array carries its own shape, so a
/// multi-component one reduces to magnitude exactly as a vector field used to.
/// What the numbers *mean* was decided by whoever bound them.
///
/// Magnitude is a defensible reduction but not always the one you want — von
/// Mises is the conventional scalar for a stress tensor — so derived quantities
/// should eventually be computed by the client and bound like anything else,
/// which is now the natural way to do it.
pub(crate) fn bound_colours(
    array: &DataArray,
    colour: &ColorBy,
    count: usize,
) -> Option<Vec<[f32; 4]>> {
    Some(
        normalised(array, colour, count)?
            .into_iter()
            .map(|t| sample(colour.map, t))
            .collect(),
    )
}

/// Where each element falls along the colour map, as 0..1.
///
/// The half of [`bound_colours`] that decides *how far along* rather than *what
/// colour*. Still split out, because a pathway that cannot read vertex colours
/// wants the position instead — to write into a texture coordinate and let a
/// ramp texture supply the colour. Nothing needs that today, so it is private.
///
/// Autoscales over the drawn elements unless `ColorBy::range` pins it.
fn normalised(array: &DataArray, colour: &ColorBy, count: usize) -> Option<Vec<f32>> {
    let values = scalars(array);
    if values.len() < count {
        return None;
    }

    let (low, high) = colour.range.unwrap_or_else(|| {
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for value in &values[..count] {
            if value.is_finite() {
                low = low.min(*value);
                high = high.max(*value);
            }
        }
        (low, high)
    });
    let span = if (high - low).abs() < f32::EPSILON {
        1.0
    } else {
        high - low
    };

    Some(
        values[..count]
            .iter()
            .map(|value| ((value - low) / span).clamp(0.0, 1.0))
            .collect(),
    )
}

/// One number per element, reducing a multi-component array to magnitude.
fn scalars(array: &DataArray) -> Vec<f32> {
    let raw = array.to_f32();
    let components = array.components().max(1) as usize;
    if components == 1 {
        return raw;
    }
    raw.chunks_exact(components)
        .map(|element| element.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// How many steps a ramp texture carries. Also the number of buckets a backend
/// quantises into when it has to colour per instance rather than per vertex, so
/// the two routes to a colour agree to within one step.
pub(crate) const RAMP_STEPS: usize = 256;

/// The colour map as a 1D image, for a pipeline that cannot read vertex colours.
///
/// `Rgba8Unorm` rather than `Rgba8UnormSrgb`: [`sample`] returns linear values,
/// and the sRGB format would have the sampler convert them a second time and
/// wash the ramp out.
///
/// Storing linear in eight bits does cost precision in the darks, which is the
/// thing sRGB encoding exists to avoid. It is not worth chasing here: a ramp is
/// [`RAMP_STEPS`] entries of smoothly varying colour with nothing fine to lose,
/// and the alternative — keeping the texture sRGB while vertex colours stay
/// linear — would mean two conventions for one colour map.
///
/// The caller must clamp to edge in both axes. Repeating wraps the top of the
/// map onto the bottom, which shows up as a hard seam at the extremes.
pub(crate) fn ramp_texture(map: ColorMap) -> Image {
    let mut data = Vec::with_capacity(RAMP_STEPS * 4);
    for step in 0..RAMP_STEPS {
        let rgba = sample(map, step as f32 / (RAMP_STEPS - 1) as f32);
        data.extend(rgba.iter().map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
    }

    let mut image = Image::new(
        Extent3d {
            width: RAMP_STEPS as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
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

/// Samples a colour map, returning **linear** RGBA.
///
/// Every colour in this project is *authored* in sRGB, because that is the
/// space colour maps are published in and the space anyone reading a hex value
/// means. Every consumer wants linear: a vertex colour reaches the shader
/// untouched and `pbr_fragment.wgsl` assigns it straight to `base_color`, and
/// [`ramp_texture`] writes into a format the hardware does not convert on read.
/// So the conversion happens here, once, at the boundary between the two — the
/// same place and the same way [`elements::colour`] does it.
///
/// This used to hand the stops back unconverted while claiming otherwise, which
/// rendered every ramp brighter and less saturated than the map it named:
/// viridis came out mid-magenta at the low end and near-white at the top rather
/// than dark purple and yellow. Correcting it made ramp-coloured actors visibly
/// darker and more saturated, which is the point rather than a side effect.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use crate::scene::data::{BufferMeta, Dtype};
    use crate::scene::registry::{ActorParams, ParamValue};
    use bevy::platform::collections::HashMap;

    fn array() -> DataArray {
        DataArray::numeric(Dtype::Float32, vec![1, 3], vec![0; 12])
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        app.init_resource::<DataStore>();
        app.add_systems(Update, mark_dirty);
        app
    }

    /// Holds one array under handle 0, and spawns a place to draw it.
    fn spawn_object(app: &mut App, name: &str) -> Entity {
        let positions = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        let meta = BufferMeta {
            name: "whatever".into(),
            dtype: Dtype::Float32,
            shape: vec![1, 3],
        };
        app.world_mut()
            .resource_mut::<DataStore>()
            .insert(0, meta, positions);
        app.world_mut()
            .spawn(SceneObject { name: name.into() })
            .id()
    }

    /// Spawns an actor drawn under `parent`.
    fn spawn_actor(app: &mut App, parent: Entity) -> Entity {
        let mut params = crate::scene::registry::ParamMap::default();
        params.insert("size".into(), ParamValue::Float(1.0));
        app.world_mut()
            .spawn((
                ActorKindId("points"),
                ActorParams(params),
                ColorBy::default(),
                Subset::All,
                Bindings(HashMap::from_iter([("positions", 0u64)])),
                ChildOf(parent),
            ))
            .id()
    }

    /// One object drawn in place — source and transform parent the same, which
    /// is what an upload produces.
    fn scene() -> (App, Entity, Entity) {
        let mut app = app();
        let object = spawn_object(&mut app, "test");
        let actor = spawn_actor(&mut app, object);
        app.update();
        (app, object, actor)
    }

    fn flags(app: &App, entity: Entity) -> Dirty {
        app.world()
            .get::<Dirty>(entity)
            .copied()
            .unwrap_or_default()
    }

    fn dirty(app: &App, entity: Entity) -> bool {
        flags(app, entity).any()
    }

    fn settle(app: &mut App, entity: Entity) {
        app.world_mut().entity_mut(entity).insert(Dirty::default());
        app.update();
        assert!(!dirty(app, entity), "should not redraw without a change");
    }

    /// Colour maps are authored in sRGB and every consumer wants linear, so
    /// [`sample`] has to convert. This went unnoticed for a long time because
    /// the failure is quiet — ramps merely looked washed out — and because
    /// [`elements::colour`] did convert, so half the renderer was right.
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

    #[test]
    fn marks_new_actors() {
        let (app, _, actor) = scene();
        assert_eq!(
            flags(&app, actor),
            Dirty::ALL,
            "an actor that has never been drawn needs everything"
        );
    }

    /// The point of grading: recolouring must not ask for a rebuild, because
    /// the vertices have not moved and a merged protein is expensive to
    /// re-tessellate.
    #[test]
    fn recolouring_does_not_ask_for_a_rebuild() {
        let (mut app, _, actor) = scene();
        settle(&mut app, actor);

        // Any change to the colouring; the ramp stands in for what used to be a
        // change of field.
        app.world_mut().get_mut::<ColorBy>(actor).unwrap().map = ColorMap::CoolWarm;
        app.update();
        assert_eq!(flags(&app, actor), Dirty::COLOUR);
    }

    #[test]
    fn redraws_when_the_dataset_changes() {
        let (mut app, _object, actor) = scene();
        settle(&mut app, actor);

        // Binding a different array is new data, not a new setting.
        app.world_mut()
            .get_mut::<Bindings>(actor)
            .unwrap()
            .0
            .insert("positions", 1);
        app.update();
        assert_eq!(flags(&app, actor), Dirty::GEOMETRY);
    }

    /// Marks accumulate rather than overwrite. Several systems classify in one
    /// tick, and an `insert` would let whichever ran last drop the rest.
    #[test]
    fn separate_reasons_accumulate() {
        let (mut app, _object, actor) = scene();
        settle(&mut app, actor);

        // Any change to the colouring; the ramp stands in for what used to be a
        // change of field.
        app.world_mut().get_mut::<ColorBy>(actor).unwrap().map = ColorMap::CoolWarm;
        app.world_mut()
            .get_mut::<Bindings>(actor)
            .unwrap()
            .0
            .insert("positions", 1);
        app.update();

        let flags = flags(&app, actor);
        assert!(flags.colour && flags.geometry, "got {flags:?}");
    }

    #[test]
    fn redraws_when_the_array_bytes_change() {
        let (mut app, _, actor) = scene();
        settle(&mut app, actor);

        // The binding still names the same handle, so only the asset event
        // reveals that the contents moved underneath it.
        let id = app
            .world()
            .resource::<DataStore>()
            .get(0)
            .unwrap()
            .handle
            .id();
        app.world_mut().write_message(AssetEvent::Modified { id });
        app.update();
        assert!(dirty(&app, actor));
    }

    #[test]
    fn ignores_arrays_the_object_does_not_hold() {
        let (mut app, _, actor) = scene();
        settle(&mut app, actor);

        // A genuinely different asset, not the one the object references.
        let other = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        app.world_mut()
            .write_message(AssetEvent::Modified { id: other.id() });
        app.update();
        assert!(!dirty(&app, actor));
    }

    /// Every actor binding an array redraws when its bytes change, wherever in
    /// the tree it sits. This used to be reached by following the source link
    /// from the object holding the data; the binding says it directly, so an
    /// actor placed under some unrelated node is no longer a special case.
    #[test]
    fn marks_every_actor_binding_an_array() {
        let mut app = app();
        let source = spawn_object(&mut app, "source");
        let elsewhere = spawn_object(&mut app, "elsewhere");
        let first = spawn_actor(&mut app, source);
        let second = spawn_actor(&mut app, elsewhere);
        app.update();
        settle(&mut app, first);
        settle(&mut app, second);

        let id = app
            .world()
            .resource::<DataStore>()
            .get(0)
            .unwrap()
            .handle
            .id();
        app.world_mut().write_message(AssetEvent::Modified { id });
        app.update();
        assert!(dirty(&app, first));
        assert!(
            dirty(&app, second),
            "placement does not decide what it reads"
        );
    }

    /// An array nothing binds marks nothing.
    #[test]
    fn leaves_actors_that_bind_something_else_alone() {
        let (mut app, _, actor) = scene();
        settle(&mut app, actor);

        let unrelated = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        app.world_mut()
            .write_message(AssetEvent::Modified { id: unrelated.id() });
        app.update();
        assert!(!dirty(&app, actor));
    }
}
