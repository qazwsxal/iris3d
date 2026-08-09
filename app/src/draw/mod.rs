//! Rendering backends.
//!
//! Each backend is a system that picks up representation entities of one kind,
//! reads the dataset of the object the representation is *of* — which need not
//! be its transform parent — and builds something drawable onto the
//! representation entity itself. Nothing in [`crate::scene`] knows these exist,
//! so a backend can be replaced or run alongside another.
//!
//! This is the straightforward `Mesh3d`-per-object baseline. It is deliberately
//! the simple option: it gets every sample dataset on screen and gives a
//! reference image to check more ambitious approaches against. It will not
//! scale — see the notes on each backend for where it gives out.

use bevy::asset::embedded_asset;
use bevy::prelude::*;

use crate::scene::data::{Field, Fields};
use crate::scene::representation::ColorMap;
use crate::scene::{
    ColorBy, DataArray, MeshData, MoleculeData, PointCloud, Representation, Representations,
    SceneObject,
};

mod molecule;
mod points;
mod surface;

pub use points::PointQuadMaterial;

/// Marks a representation entity whose drawable output is out of date.
///
/// Backends consume this rather than `Added<Representation>`, so a change to
/// the representation, its colouring, the dataset, or the underlying array
/// bytes all produce a redraw — not just the initial upload.
#[derive(Component)]
pub struct NeedsRedraw;

/// Ordering label for the backends, so dirty marking runs before them and
/// clearing runs after.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct Backends;

pub struct DrawPlugin;

impl Plugin for DrawPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "point_quad.wgsl");
        app.add_plugins(MaterialPlugin::<PointQuadMaterial>::default())
            .add_systems(
                Update,
                (
                    // Representations are spawned by the scene during Update,
                    // so this whole chain runs after it to pick them up on the
                    // frame they appear.
                    mark_dirty,
                    (
                        points::draw_points,
                        surface::draw_surfaces,
                        molecule::draw_molecules,
                    )
                        .in_set(Backends),
                    clear_dirty,
                )
                    .chain()
                    .after(crate::scene::apply_scene_commands),
            );
    }
}

/// Flags representations that need rebuilding.
fn mark_dirty(
    mut commands: Commands,
    changed_representations: Query<
        Entity,
        (
            With<Representation>,
            Or<(Changed<Representation>, Changed<ColorBy>)>,
        ),
    >,
    changed_datasets: Query<
        &Representations,
        Or<(
            Changed<PointCloud>,
            Changed<MeshData>,
            Changed<MoleculeData>,
            Changed<Fields>,
        )>,
    >,
    mut array_events: MessageReader<AssetEvent<DataArray>>,
    objects: Query<(&SceneObject, &Representations)>,
) {
    for entity in &changed_representations {
        commands.entity(entity).insert(NeedsRedraw);
    }

    // Following the source link rather than the child list is what makes this
    // correct for shared data: a representation parented under some other node
    // still redraws when the object it actually draws changes.
    for drawn_by in &changed_datasets {
        for representation in drawn_by.iter() {
            commands.entity(representation).insert(NeedsRedraw);
        }
    }

    // Array contents can be rewritten without the components referring to them
    // changing at all, so watch the assets directly.
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
    for (object, drawn_by) in &objects {
        let touched = object
            .arrays
            .iter()
            .any(|array| modified.contains(&array.handle.id()));
        if !touched {
            continue;
        }
        for representation in drawn_by.iter() {
            commands.entity(representation).insert(NeedsRedraw);
        }
    }
}

/// Clears the flag once every backend has had a chance at it.
///
/// Done centrally rather than per-backend because a representation is only
/// handled by the one backend that understands it, and the others must not
/// clear a flag they ignored.
fn clear_dirty(mut commands: Commands, dirty: Query<Entity, With<NeedsRedraw>>) {
    for entity in &dirty {
        commands.entity(entity).remove::<NeedsRedraw>();
    }
}

/// Resolves the field a representation is coloured by.
///
/// `None` means flat, and nothing is inferred here. An earlier version fell
/// back to the first scalar field when none was set, which meant the render was
/// coloured by a field the UI reported as "flat". The default is now chosen
/// once, explicitly, when the representation is created — see
/// `scene::default_colour_field`.
pub(crate) fn colour_field<'a>(colour: &ColorBy, fields: Option<&'a Fields>) -> Option<&'a Field> {
    // A name that no longer resolves means the field went away; fall back to
    // flat rather than silently picking a different one.
    fields?.0.get(colour.field.as_ref()?)
}

/// Reduces a field to one number per element.
///
/// Colour mapping needs a scalar, so vector and tensor fields are reduced to
/// their magnitude. That is a defensible default but not always the one you
/// want — von Mises is the conventional scalar for a stress tensor, for
/// instance — so derived quantities should eventually be selectable in their
/// own right rather than assumed here.
pub(crate) fn scalarise(field: &Field, array: &DataArray) -> Vec<f32> {
    let raw = array.to_f32();
    let components = field.meta.components().max(1) as usize;
    if components == 1 {
        return raw;
    }
    raw.chunks_exact(components)
        .map(|element| element.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// Maps a field's values onto vertex colours, autoscaling unless a range is set.
pub(crate) fn vertex_colours(
    field: &Field,
    colour: &ColorBy,
    arrays: &Assets<DataArray>,
    count: usize,
) -> Option<Vec<[f32; 4]>> {
    let values = scalarise(field, arrays.get(&field.array)?);
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
            &[[0.230, 0.299, 0.754], [0.865, 0.865, 0.865], [0.706, 0.016, 0.150]],
            t,
        ),
        ColorMap::Grayscale => [t, t, t],
        // Element colouring is per-atom, not a ramp; molecules handle it
        // directly and never reach here.
        ColorMap::ByElement => [0.8, 0.8, 0.85],
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::data::{BufferMeta, Dtype, NamedArray};
    use crate::scene::RepresentationOf;

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
        app.add_systems(Update, mark_dirty);
        app
    }

    /// Spawns an object holding one point-cloud dataset.
    fn spawn_object(app: &mut App, name: &str) -> Entity {
        let positions = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        app.world_mut()
            .spawn((
                SceneObject {
                    name: name.into(),
                    arrays: vec![NamedArray {
                        meta: BufferMeta {
                            name: "positions".into(),
                            dtype: Dtype::Float32,
                            shape: vec![1, 3],
                        },
                        handle: positions.clone(),
                    }],
                },
                PointCloud { positions },
            ))
            .id()
    }

    /// Spawns a representation drawing `source`, placed under `parent`.
    fn spawn_representation(app: &mut App, source: Entity, parent: Entity) -> Entity {
        app.world_mut()
            .spawn((
                Representation::Points { size: 1.0 },
                ColorBy::default(),
                RepresentationOf(source),
                ChildOf(parent),
            ))
            .id()
    }

    /// One object drawn in place — source and transform parent the same, which
    /// is what an upload produces.
    fn scene() -> (App, Entity, Entity) {
        let mut app = app();
        let object = spawn_object(&mut app, "test");
        let representation = spawn_representation(&mut app, object, object);
        app.update();
        (app, object, representation)
    }

    fn dirty(app: &App, entity: Entity) -> bool {
        app.world().get::<NeedsRedraw>(entity).is_some()
    }

    fn settle(app: &mut App, entity: Entity) {
        app.world_mut().entity_mut(entity).remove::<NeedsRedraw>();
        app.update();
        assert!(!dirty(app, entity), "should not redraw without a change");
    }

    #[test]
    fn marks_new_representations() {
        let (app, _, representation) = scene();
        assert!(dirty(&app, representation));
    }

    #[test]
    fn redraws_when_the_representation_changes() {
        let (mut app, _, representation) = scene();
        settle(&mut app, representation);

        *app.world_mut()
            .get_mut::<Representation>(representation)
            .unwrap() = Representation::Points { size: 5.0 };
        app.update();
        assert!(dirty(&app, representation));
    }

    #[test]
    fn redraws_when_colouring_changes() {
        let (mut app, _, representation) = scene();
        settle(&mut app, representation);

        app.world_mut()
            .get_mut::<ColorBy>(representation)
            .unwrap()
            .field = Some("stress".into());
        app.update();
        assert!(dirty(&app, representation));
    }

    #[test]
    fn redraws_when_the_dataset_changes() {
        let (mut app, object, representation) = scene();
        settle(&mut app, representation);

        // Obtaining a `Mut` is not enough — change ticks are written on deref.
        app.world_mut()
            .get_mut::<PointCloud>(object)
            .unwrap()
            .set_changed();
        app.update();
        assert!(dirty(&app, representation));
    }

    #[test]
    fn redraws_when_the_array_bytes_change() {
        let (mut app, _, representation) = scene();
        settle(&mut app, representation);

        // The component still points at the same handle, so only the asset
        // event reveals that the contents moved underneath it.
        let id = app
            .world()
            .get::<SceneObject>(app.world().entity(object_of(&app, representation)).id())
            .unwrap()
            .arrays[0]
            .handle
            .id();
        app.world_mut().write_message(AssetEvent::Modified { id });
        app.update();
        assert!(dirty(&app, representation));
    }

    #[test]
    fn ignores_arrays_the_object_does_not_hold() {
        let (mut app, _, representation) = scene();
        settle(&mut app, representation);

        // A genuinely different asset, not the one the object references.
        let other = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(array());
        app.world_mut().write_message(AssetEvent::Modified {
            id: other.id(),
        });
        app.update();
        assert!(!dirty(&app, representation));
    }

    /// The object a representation *draws*, which since the source and transform
    /// links were separated is no longer the same question as its parent.
    fn object_of(app: &App, representation: Entity) -> Entity {
        app.world()
            .get::<RepresentationOf>(representation)
            .expect("representation has a source")
            .0
    }

    /// The whole point of the split: data comes from the source, not the
    /// transform parent, so one dataset can be drawn at another node's
    /// placement. Before this, `mark_dirty` walked the parent's children and
    /// such a representation would never have been marked at all.
    #[test]
    fn redraws_from_its_source_not_its_parent() {
        let mut app = app();
        let source = spawn_object(&mut app, "source");
        let elsewhere = spawn_object(&mut app, "elsewhere");
        let representation = spawn_representation(&mut app, source, elsewhere);
        app.update();
        settle(&mut app, representation);

        app.world_mut()
            .get_mut::<PointCloud>(elsewhere)
            .unwrap()
            .set_changed();
        app.update();
        assert!(
            !dirty(&app, representation),
            "the transform parent's data is not what is being drawn"
        );

        app.world_mut()
            .get_mut::<PointCloud>(source)
            .unwrap()
            .set_changed();
        app.update();
        assert!(dirty(&app, representation), "the source's data is");
    }

    /// Two representations of one object redraw together, which is the case
    /// that could not be expressed at all before.
    #[test]
    fn marks_every_representation_of_an_object() {
        let mut app = app();
        let source = spawn_object(&mut app, "source");
        let first = spawn_representation(&mut app, source, source);
        let second = spawn_representation(&mut app, source, source);
        app.update();
        settle(&mut app, first);
        settle(&mut app, second);

        app.world_mut()
            .get_mut::<PointCloud>(source)
            .unwrap()
            .set_changed();
        app.update();
        assert!(dirty(&app, first));
        assert!(dirty(&app, second));
    }
}

fn ramp(stops: &[[f32; 3]], t: f32) -> [f32; 3] {
    let scaled = t * (stops.len() - 1) as f32;
    let index = (scaled.floor() as usize).min(stops.len() - 2);
    let frac = scaled - index as f32;
    let (a, b) = (stops[index], stops[index + 1]);
    [
        a[0] + (b[0] - a[0]) * frac,
        a[1] + (b[1] - a[1]) * frac,
        a[2] + (b[2] - a[2]) * frac,
    ]
}
