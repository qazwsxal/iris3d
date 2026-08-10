//! Rendering backends.
//!
//! Each backend is a system that picks up actor entities of one kind, reads the
//! arrays that actor has *bound* — not whatever the node it hangs under happens
//! to hold — and builds something drawable onto the actor entity itself. Nothing
//! in [`crate::scene`] knows these exist, so a backend can be replaced or run
//! alongside another.
//!
//! What each kind needs is declared, not assumed: a kind states its inputs as
//! array parameters, so a client is told that `points` wants `float32 [n, 3]`
//! rather than having to name an array "positions" and hope.
//!
//! This is the straightforward `Mesh3d`-per-actor baseline. It is deliberately
//! the simple option: it gets every sample dataset on screen and gives a
//! reference image to check more ambitious approaches against. It will not
//! scale — see the notes on each backend for where it gives out.

use bevy::asset::embedded_asset;
use bevy::prelude::*;

use crate::scene::actor::ColorMap;

use crate::scene::link::Placement;
use crate::scene::registry::{ActorKindId, ActorRegistry, Bindings};
use crate::scene::{ColorBy, DataArray, DataStore, Subset};

mod molecule;
pub mod moment;
mod points;
mod surface;
mod volume;

pub use points::PointQuadMaterial;
pub use volume::VolumeMaterial;

/// What about an actor's drawable output is out of date.
///
/// Graded rather than a single flag, because the three differ by orders of
/// magnitude in cost. Re-tessellating a protein to drag a colour-map slider
/// meant rebuilding a merged mesh of tens of thousands of vertices per frame to
/// change four bytes each.
///
/// Flags accumulate and are cleared together once every backend has had its
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
/// tick — the generic classifier and each backend's own — and an `insert` would
/// let whichever ran last drop the others' findings. `or_default` also means no
/// backend has to arrange for the component to exist first.
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

/// What every backend needs to redraw one actor: its own style, how it is
/// coloured, whose data it draws, what is out of date, and whatever it
/// produced last time — which is what makes reuse rather than reallocation
/// possible.
pub(crate) type Drawable<'a, Style, Material> = (
    Entity,
    &'a Style,
    &'a ColorBy,
    &'a Subset,
    &'a Bindings,
    &'a Dirty,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<Material>>,
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
/// of them has marked before any backend reads the result.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Invalidate;

/// Ordering label for the backends, so dirty marking runs before them and
/// clearing runs after.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct Backends;

/// Replaces a mesh in place when the actor already has one.
///
/// Reusing the handle keeps the entity pointing at the same asset — and, more
/// to the point, stops every rebuild leaking a fresh `Mesh` into `Assets`.
pub(crate) fn ensure_mesh(
    commands: &mut Commands,
    entity: Entity,
    meshes: &mut Assets<Mesh>,
    existing: Option<&Mesh3d>,
    mesh: Mesh,
) {
    if let Some(Mesh3d(handle)) = existing
        && let Some(mut slot) = meshes.get_mut(handle)
    {
        *slot = mesh;
        return;
    }
    commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
}

/// As [`ensure_mesh`], for the material.
pub(crate) fn ensure_material<M: Material>(
    commands: &mut Commands,
    entity: Entity,
    materials: &mut Assets<M>,
    existing: Option<&MeshMaterial3d<M>>,
    material: M,
) {
    if let Some(MeshMaterial3d(handle)) = existing
        && let Some(mut slot) = materials.get_mut(handle)
    {
        *slot = material;
        return;
    }
    commands
        .entity(entity)
        .insert(MeshMaterial3d(materials.add(material)));
}

/// Overwrites a mesh's vertex colours without touching anything else.
///
/// Only legal when the vertex count is unchanged, which is exactly when the
/// geometry is not also dirty.
pub(crate) fn repaint(
    meshes: &mut Assets<Mesh>,
    existing: Option<&Mesh3d>,
    colours: Vec<[f32; 4]>,
) {
    let Some(Mesh3d(handle)) = existing else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(handle) else {
        return;
    };
    if mesh.count_vertices() != colours.len() {
        // Should not happen, and silently painting a prefix would be worse than
        // waiting for the rebuild that is evidently coming.
        warn!(
            "draw: {} vertex colours for a mesh of {} vertices; skipping the repaint",
            colours.len(),
            mesh.count_vertices()
        );
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "point_quad.wgsl");
        embedded_asset!(app, "volume.wgsl");

        // Declaring the kinds is what makes them exist at all — `scene` holds
        // no list of its own. `init_resource` rather than `insert_resource` so
        // a second backend plugin can register alongside this one whichever
        // order they are added in. Registration order decides which kind an
        // upload of each dataset shape is drawn with.
        app.init_resource::<ActorRegistry>();
        {
            let mut registry = app.world_mut().resource_mut::<ActorRegistry>();
            points::register(&mut registry);
            surface::register(&mut registry);
            molecule::register(&mut registry);
            volume::register(&mut registry);
        }

        // No representation kind yet, on purpose: the moment backend's
        // render-world half is independent of the registry, and the registry is
        // being reshaped. Registering a kind is the last step, not the first.
        app.add_plugins(moment::MomentPlugin);

        app.add_plugins((
            MaterialPlugin::<PointQuadMaterial>::default(),
            MaterialPlugin::<VolumeMaterial>::default(),
        ))
        .add_systems(
            Update,
            (
                // Actors are spawned by the scene during Update, so this whole
                // chain runs after it to pick them up on the frame they
                // appear.
                //
                // Each backend classifies its own style changes: only it
                // knows whether one of its parameters feeds the geometry or
                // a material uniform, and centralising that would put
                // backend knowledge back in shared code.
                (
                    mark_dirty,
                    points::invalidate,
                    surface::invalidate,
                    molecule::invalidate,
                    volume::invalidate,
                )
                    .in_set(Invalidate),
                (
                    points::draw_points,
                    surface::draw_surfaces,
                    molecule::draw_molecules,
                    volume::draw_volumes,
                )
                    .in_set(Backends),
                // After the backends, so a placement picks up a handle on the
                // same frame the actor gets one rather than a frame late.
                //
                // No `VolumeMaterial` here: a volume's uniform holds the world
                // transform of the copy being drawn, so its placements each get
                // their own material rather than a copy of one. See `volume`.
                (
                    copy_meshes,
                    copy_materials::<StandardMaterial>,
                    copy_materials::<PointQuadMaterial>,
                ),
                clear_dirty,
            )
                .chain()
                // Style components are derived from the parameters, so an
                // actor has no style at all until that has run.
                .after(crate::scene::registry::apply_actor_params),
        );
    }
}

/// Gives every placement of an actor the mesh that actor owns.
///
/// The handle, not the geometry. A backend rebuilds into the asset it already
/// holds, so the copies never need touching again — this only has to run when
/// an actor's handle is first created or genuinely replaced, and the comparison
/// makes that the common case. Several placements sharing one mesh and one
/// material is also what lets Bevy batch them into a single draw.
fn copy_meshes(
    mut commands: Commands,
    actors: Query<&Mesh3d>,
    placements: Query<(Entity, &Placement, Option<&Mesh3d>)>,
) {
    for (entity, placement, current) in &placements {
        let Ok(mesh) = actors.get(placement.0) else {
            // The actor has not been drawn yet. Nothing to copy, and it will be
            // here next frame.
            continue;
        };
        if current.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert(mesh.clone());
        }
    }
}

/// As [`copy_meshes`], for the material.
///
/// Generic because the material type is the backend's choice, so this is
/// registered once per material the build knows about.
fn copy_materials<M: Material>(
    mut commands: Commands,
    actors: Query<&MeshMaterial3d<M>>,
    placements: Query<(Entity, &Placement, Option<&MeshMaterial3d<M>>)>,
) {
    for (entity, placement, current) in &placements {
        let Ok(material) = actors.get(placement.0) else {
            continue;
        };
        if current.map(|MeshMaterial3d(handle)| handle.id()) != Some(material.0.id()) {
            commands.entity(entity).insert(material.clone());
        }
    }
}

/// Flags what needs redoing, for the reasons any backend would agree on.
///
/// Style parameters are not among them — what a parameter affects is the
/// backend's business, so each classifies its own. See the `invalidate` system
/// in each of them.
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

/// Clears the flags once every backend has had a chance at them.
///
/// Done centrally rather than per-backend because an actor is only handled by
/// the one backend that understands it, and the others must not clear flags
/// they ignored. Cleared in place rather than removed, so the component stays
/// put and marking never costs an archetype move.
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
    let raw = array.to_f32();
    let components = array.components().max(1) as usize;
    let values: Vec<f32> = if components == 1 {
        raw
    } else {
        raw.chunks_exact(components)
            .map(|element| element.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect()
    };
    scale_into_map(&values, colour, count)
}

/// Scales values into the colour map, autoscaling unless a range is set.
fn scale_into_map(values: &[f32], colour: &ColorBy, count: usize) -> Option<Vec<[f32; 4]>> {
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
            .map(|value| sample(colour.map, ((value - low) / span).clamp(0.0, 1.0)))
            .collect(),
    )
}

/// Nine evenly spaced stops, linearly interpolated. Enough to be perceptually
/// honest without carrying a 256-entry table.
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
/// Colour-map stops are quoted in sRGB, as they are everywhere else, but vertex
/// colours reach the shader unconverted — `pbr_fragment.wgsl` assigns them
/// directly to `base_color`. Handing over sRGB values renders everything washed
/// out, so convert here, once.
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
    [rgb[0], rgb[1], rgb[2], 1.0]
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
        DataArray {
            dtype: Dtype::Float32,
            shape: vec![1, 3],
            data: vec![0; 12],
        }
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
