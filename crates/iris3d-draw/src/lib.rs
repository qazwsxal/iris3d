//! Rendering backends.
//!
//! A **backend** is a whole rendering pathway: one pipeline, together with the
//! actor kinds built for it. Backends are mutually exclusive, and which one
//! runs is decided once at launch. Two techniques that composite differently
//! cannot share a frame correctly, so choosing once removes the whole class of
//! interop questions rather than answering them one at a time.
//!
//! [`default`](mod@default) is the only pathway built. It accumulates moments:
//! opaque geometry goes through Bevy's ordinary passes, and anything
//! transmitting deposits absorbance into a shared buffer instead of blending,
//! so a structure inside the density map it was built from composes correctly
//! with nothing sorted.
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
//! A pathway that cannot run on this machine refuses rather than substituting
//! another — see [`probe`]. Why the seam is kept with one pathway behind it,
//! and what was tried before, is in `docs/design/backends.md`.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::settings::WgpuFeatures;

use bevy::color::ColorToComponents;

use iris3d_filter::colormap::{ColorMap, sample};
use iris3d_model::{Bindings, ParamKind, ParamMap, ParamSpec};
use iris3d_scene::registry::{ActorKindId, ActorRegistry};
use iris3d_scene::{DataArray, DataStore};

pub mod atoms;
pub mod default;
pub mod glycan;
pub mod probe;
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

/// What every actor has, whichever backend is running: its own style, how much
/// of the data it draws, what it draws, and what is out of date.
///
/// Colouring is not here. It is a filter, so a colour reaches an actor as an
/// ordinary bound array and needs no place of its own — and a flat colour is a
/// parameter like any other, living in the kind's own style component.
///
/// A backend extends this with whatever *it* produced last time, which is what
/// makes reuse rather than reallocation possible — and is precisely the part
/// that depends on the pipeline. Each kind declares its own; see the `Drawable` alias in any of them.
pub(crate) type Actor<'a, Style> = (Entity, &'a Style, &'a Bindings, &'a Dirty);

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
    arrays.get(&store.array(bindings.get(input)?)?.handle)
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
                Invalidate.after(iris3d_scene::registry::apply_actor_params),
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
    registry: Res<ActorRegistry>,
    new_actors: Query<Entity, Added<ActorKindId>>,
    rebound: Query<Entity, (With<ActorKindId>, Changed<Bindings>)>,
    mut array_events: MessageReader<AssetEvent<DataArray>>,
    mut mesh_events: MessageReader<AssetEvent<Mesh>>,
    store: Res<DataStore>,
    bindings: Query<(Entity, &ActorKindId, &Bindings)>,
) {
    for entity in &new_actors {
        mark(&mut commands, entity, Dirty::ALL);
    }

    // Binding a different array is new data, not a new setting: the vertex count
    // itself changes, so there is nothing to write in place.
    //
    // Not graded by `structural`, unlike the asset path below. `Changed` is per
    // component and `Bindings` is one component, so this says *something* was
    // rebound without saying which input — and an actor whose colour moved has
    // its positions bound too. Grading it would mean keeping the previous map to
    // diff against, to save a rebuild on an operation that happens once when a
    // scene is built rather than every frame a slider moves.
    for entity in &rebound {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }

    // A geometry filter finishing rewrites a `Mesh` an actor already holds, and
    // Bevy re-uploads it with nothing here involved — so this is *not* how the
    // new vertices reach the screen. What it is for is the one thing about an
    // actor that depends on the mesh's contents rather than its identity:
    // whether it carries colours, which decides whether `surface` uses its flat
    // tint. A rebuild would be wrong; there is nothing to rebuild.
    let remeshed: Vec<_> = mesh_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    if !remeshed.is_empty() {
        for (actor, _, bound) in &bindings {
            let touched = bound.0.values().any(|handle| {
                store
                    .geometry(*handle)
                    .is_some_and(|held| remeshed.contains(&held.handle.id()))
            });
            if touched {
                mark(&mut commands, actor, Dirty::MATERIAL);
            }
        }
    }

    // Array contents can be rewritten without any binding changing, so watch the
    // assets directly. This is the path a filter finishing takes.
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
    for (actor, kind, bound) in &bindings {
        let changed = |handle: u64| {
            store
                .array(handle)
                .is_some_and(|held| modified.contains(&held.handle.id()))
        };
        let what = invalidated(&registry, kind, bound, changed);
        if what.any() {
            mark(&mut commands, actor, what);
        }
    }
}

/// What a change to some of an actor's bound arrays makes out of date.
///
/// `Dirty::GEOMETRY` if any changed array feeds a `structural` input, and
/// `Dirty::COLOUR` if the changed ones are all non-structural. Nothing at all if
/// none of them changed.
///
/// This is what keeps a colour-map drag cheap now that colouring is a filter.
/// The filter rewrites its output array, which reaches an actor as an ordinary
/// asset change — indistinguishable, without the declaration, from someone
/// re-uploading its positions.
fn invalidated(
    registry: &ActorRegistry,
    kind: &ActorKindId,
    bound: &Bindings,
    changed: impl Fn(u64) -> bool,
) -> Dirty {
    // A kind nothing registered has no draw system either, so there is nothing
    // for a flag to reach. `apply_actor_params` is where that gets reported.
    let Some(registered) = registry.get(kind.0) else {
        return Dirty::default();
    };
    let mut what = Dirty::default();
    for (input, handle) in &bound.0 {
        if !changed(*handle) {
            continue;
        }
        // An input the kind does not declare cannot be read, so it cannot have
        // invalidated anything.
        let Some(ParamKind::Array { structural, .. }) = registered.spec(input).map(|s| s.kind)
        else {
            continue;
        };
        let reason = match structural {
            true => Dirty::GEOMETRY,
            false => Dirty::COLOUR,
        };
        what.geometry |= reason.geometry;
        what.colour |= reason.colour;
    }
    what
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

/// Reads a bound colour array as vertex colours.
///
/// The array is **already linear RGB**, one triple per element. Nothing here
/// maps, scales or reduces: how numbers became colours was decided by whatever
/// wrote the array, which for a scalar field is the `colormap` filter.
///
/// Choosing a ramp here instead would put "which ramp" in the same place as
/// "how to rasterise", and leave an actor colourable exactly one way. See
/// [`iris3d_filter::colormap`].
///
/// `None` when the array is too short for the elements being drawn, which is the
/// honest answer to a colour array that does not match its positions: better an
/// untinted mesh than one whose colours are offset from its vertices.
pub(crate) fn bound_colours(array: &DataArray, count: usize) -> Option<Vec<[f32; 4]>> {
    if array.components() != 3 {
        return None;
    }
    let values = array.to_f32();
    if values.len() < count * 3 {
        return None;
    }
    Some(
        values[..count * 3]
            .chunks_exact(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2], 1.0])
            .collect(),
    )
}

/// A colour parameter, as **linear** RGB.
///
/// Declared in sRGB, because that is the space a person picking a colour means
/// and the space a hex value is written in. Converted once, here, so nothing
/// downstream has to remember which it is holding — the same boundary
/// [`iris3d_filter::colormap::sample`] converts at.
pub(crate) fn tint(params: &ParamMap, id: &str, fallback: Vec3) -> Vec3 {
    let srgb = iris3d_model::vec3(params, id, fallback);
    Color::srgb(srgb.x, srgb.y, srgb.z).to_linear().to_vec3()
}

/// What an unbound colour falls back to: a pale neutral that reads as "not
/// coloured by anything" rather than as a choice.
pub(crate) const UNTINTED: &[f64] = &[0.8, 0.8, 0.85];

/// The flat colour every kind offers, declared once.
///
/// Shared because it means the same thing everywhere, and because a kind
/// spelling its own range or default differently would be a bug rather than a
/// choice. What it *does* still differs by kind — a lit surface takes it as a base
/// colour and a medium as a transmission — and each says so where it reads it.
pub(crate) const TINT: ParamSpec = ParamSpec {
    id: "tint",
    label: "colour",
    kind: ParamKind::Vector {
        components: 3,
        default: UNTINTED,
        min: 0.0,
        max: 1.0,
        integral: false,
    },
};

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
        data.extend(
            rgba.iter()
                .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::platform::collections::HashMap;
    use iris3d_data::array::{BufferMeta, Dtype};
    use iris3d_model::ParamValue;
    use iris3d_scene::SceneObject;
    use iris3d_scene::registry::ActorParams;

    fn array() -> DataArray {
        DataArray::numeric(Dtype::Float32, vec![1, 3], vec![0; 12])
    }

    /// One structural input and one that only repaints, which is the whole
    /// distinction `mark_dirty` now reads.
    const SPECS: &[iris3d_model::ParamSpec] = &[
        iris3d_model::ParamSpec {
            id: "positions",
            label: "positions",
            kind: ParamKind::Array {
                dtypes: &[Dtype::Float32],
                shape: &[0, 3],
                required: true,
                structural: true,
            },
        },
        iris3d_model::ParamSpec {
            id: "colour",
            label: "colour",
            kind: ParamKind::Array {
                dtypes: &[],
                shape: &[0],
                required: false,
                structural: false,
            },
        },
    ];

    fn app() -> App {
        let mut app = App::new();
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        // Geometry is an asset like any other, and `mark_dirty` watches it: a
        // filter rewriting a mesh has to reach the actors drawing it.
        app.add_message::<AssetEvent<Mesh>>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<DataStore>();
        app.init_resource::<ActorRegistry>();
        // `mark_dirty` asks the registry what a changed array invalidates, so a
        // kind that is not registered would be marked for nothing at all.
        app.world_mut().resource_mut::<ActorRegistry>().register(
            iris3d_scene::registry::ActorKind {
                id: "points",
                label: "points",
                params: SPECS,
                apply: |_, _| {},
            },
        );
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
        let mut params = iris3d_model::ParamMap::default();
        params.insert("size".into(), ParamValue::Float(1.0));
        app.world_mut()
            .spawn((
                ActorKindId("points"),
                ActorParams(params),
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

    /// Binds a colour array and rewrites it, which is what a `colormap` filter
    /// finishing looks like from here: the binding does not move, the bytes
    /// behind it do.
    fn recolour(app: &mut App, actor: Entity) {
        let existing = app
            .world()
            .resource::<DataStore>()
            .array(1)
            .map(|held| held.handle.id());
        let id = match existing {
            Some(id) => id,
            None => {
                let handle = app
                    .world_mut()
                    .resource_mut::<Assets<DataArray>>()
                    .add(array());
                let id = handle.id();
                let meta = BufferMeta {
                    name: "colour".into(),
                    dtype: Dtype::Float32,
                    shape: vec![1, 3],
                };
                app.world_mut()
                    .resource_mut::<DataStore>()
                    .insert(1, meta, handle);
                app.world_mut()
                    .get_mut::<Bindings>(actor)
                    .unwrap()
                    .0
                    .insert("colour", 1u64);
                app.update();
                settle(app, actor);
                id
            }
        };
        app.world_mut().write_message(AssetEvent::Modified { id });
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

        // Rewriting the array bound to the non-structural `colour` input, which
        // is what a `colormap` filter finishing looks like from here.
        recolour(&mut app, actor);
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

        recolour(&mut app, actor);
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
            .array(0)
            .unwrap()
            .handle
            .id();
        app.world_mut().write_message(AssetEvent::Modified { id });
        app.update();
        assert_eq!(
            flags(&app, actor),
            Dirty::GEOMETRY,
            "positions are structural, so nothing survives"
        );
    }

    /// The reason `structural` is declared at all.
    ///
    /// Colouring is a filter now, so a colour-map drag reaches an actor as an
    /// ordinary asset change — the same event a re-upload of its positions
    /// produces. Without the input's own declaration the two are
    /// indistinguishable, and every drag would re-tessellate the mesh to change
    /// three floats a vertex.
    #[test]
    fn rewriting_a_colour_array_repaints_rather_than_rebuilds() {
        let (mut app, _, actor) = scene();

        // A second array, bound to the non-structural input.
        let colours = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        let id = colours.id();
        let meta = BufferMeta {
            name: "colour".into(),
            dtype: Dtype::Float32,
            shape: vec![1, 3],
        };
        app.world_mut()
            .resource_mut::<DataStore>()
            .insert(1, meta, colours);
        app.world_mut()
            .get_mut::<Bindings>(actor)
            .unwrap()
            .0
            .insert("colour", 1u64);
        // The rebind itself dirties the actor, so let that land before asking
        // what the *contents* changing does.
        app.update();
        settle(&mut app, actor);

        app.world_mut().write_message(AssetEvent::Modified { id });
        app.update();
        assert_eq!(flags(&app, actor), Dirty::COLOUR);
    }

    /// Rebinding is *not* graded, deliberately. `Bindings` is one component, so
    /// a change to it says nothing about which input moved, and an actor whose
    /// colour was rebound has its positions bound too.
    #[test]
    fn rebinding_anything_asks_for_a_rebuild() {
        let (mut app, _, actor) = scene();
        settle(&mut app, actor);

        app.world_mut()
            .get_mut::<Bindings>(actor)
            .unwrap()
            .0
            .insert("colour", 0u64);
        app.update();
        assert_eq!(flags(&app, actor), Dirty::GEOMETRY);
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
    /// the tree it sits. The binding says so directly, so an actor placed under
    /// some unrelated node is not a special case.
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
            .array(0)
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
