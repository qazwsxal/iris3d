//! Does a backend's plugin actually stand up with each of the kinds it
//! registers?
//!
//! Every other test in `draw` builds its systems by hand: a kind's own module
//! adds `mark_dirty`, its `invalidate` and its `draw_*` to a bare `App`, inits
//! exactly the assets that kind touches, and asserts about the meshes that come
//! out. That is the right shape for testing what a kind *produces*, and it is
//! blind to a whole class of bug — because the thing being tested is a
//! hand-assembled imitation of the plugin rather than the plugin.
//!
//! `points` is the worked example. Its kind was registered and its systems were
//! added, but `MaterialPlugin::<PointQuadMaterial>` was not, so the
//! `ResMut<Assets<PointQuadMaterial>>` its draw system asks for did not exist.
//! Bevy fails that parameter validation by panicking, so the app died on the
//! first frame a point cloud existed. Its own tests passed throughout: they
//! called `init_resource::<Assets<PointQuadMaterial>>()` themselves.
//!
//! So this builds the **real** backend plugin, spawns one actor of each kind the
//! registry reports, and runs the schedule. What is asserted afterwards is
//! deliberately thin — the test's content is that the frames happened at all.
//! A kind whose systems cannot get their parameters brings the app down here
//! rather than in front of a user.
//!
//! # What it does not cover
//!
//! No GPU. The render app never exists, so `finish` is never called and nothing
//! here reaches a pipeline, a shader or a pixel. A backend can pass this and
//! still fail to draw. It covers the main-world half: registration, parameter
//! application, binding, invalidation, the draw systems and the placement copy.
//!
//! The arrays are synthesised from what each input *declares* — first accepted
//! dtype, declared shape with `n` filled in, zero bytes — rather than being real
//! data. That is the point: it means a kind added later is covered without
//! editing this file, and it means each kind meets plausible-but-degenerate
//! input, which is what a client can send.

use bevy::prelude::*;

use crate::scene::data::{BufferMeta, Dtype};
use crate::scene::link::{Parents, Shown};
use crate::scene::registry::{
    ActorKindId, ActorParams, ActorRegistry, ParamKind, ParamSpec, ParamValue,
};
use crate::scene::{DataArray, DataStore, SceneObject, Subset};

use super::{Backend, DrawPlugin};

/// Elements in every synthesised array.
///
/// Small enough to keep the test instant, big enough that a kind reading
/// triples, pairs or neighbouring residues has something to read.
const ELEMENTS: u64 = 8;

/// The app a backend plugin is built into.
///
/// `MinimalPlugins` plus the asset types a backend expects to find already
/// registered — in the real app those come from `DefaultPlugins`, which needs a
/// window and an adapter. Nothing here stands in for the render app: the
/// backend's `finish` is never called, because `App::update` does not call it.
fn headless(backend: Backend) -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()));
    app.init_asset::<Shader>()
        .init_asset::<Mesh>()
        .init_asset::<Image>()
        .init_asset::<StandardMaterial>();

    // The scene's half of what a backend reads. Not `ScenePlugin`, which also
    // drains the gRPC command queue and so wants a bridge; these two systems are
    // the part that turns a spawned actor into something drawable.
    app.init_asset::<DataArray>()
        .init_resource::<DataStore>()
        .init_resource::<crate::redraw::KeepAwake>()
        .add_systems(
            Update,
            (
                crate::scene::registry::apply_actor_params,
                crate::scene::link::sync_placements,
            )
                .chain(),
        );

    app.add_plugins(DrawPlugin { backend });
    app
}

/// An array that satisfies what an input declared.
///
/// Read off the [`ParamKind::Array`] rather than written per kind, so this stays
/// correct for inputs that do not exist yet. A `0` in the declared shape is the
/// free axis and becomes [`ELEMENTS`]; a fixed axis is taken as given.
fn conforming(spec: &ParamSpec, id: u64, app: &mut App) -> Option<u64> {
    let ParamKind::Array { dtypes, shape, .. } = spec.kind else {
        return None;
    };
    // The first accepted type. An empty list accepts anything, and `float32` is
    // what a client would most likely send.
    let dtype = dtypes.first().copied().unwrap_or(Dtype::Float32);
    let shape: Vec<u64> = if shape.is_empty() {
        vec![ELEMENTS]
    } else {
        shape
            .iter()
            .map(|axis| if *axis == 0 { ELEMENTS } else { *axis })
            .collect()
    };
    let count: u64 = shape.iter().product();

    let array = if dtype == Dtype::Str {
        // Text lives in `strings`, one entry per element, and `data` stays
        // empty — see `DataArray::strings`. `CA` because the one text input
        // there is names atoms, and a backbone builder that finds none does
        // nothing rather than exercising anything.
        DataArray {
            dtype,
            shape: shape.clone(),
            data: Vec::new(),
            strings: vec!["CA".to_string(); count as usize],
        }
    } else {
        // Zeros: valid for every numeric type, and every index in range
        // whatever it points into. Degenerate geometry is a legitimate thing
        // for a client to send, so a kind that cannot survive it is worth
        // hearing about.
        DataArray::numeric(
            dtype,
            shape.clone(),
            vec![0u8; (count * dtype.size()) as usize],
        )
    };

    let handle = app
        .world_mut()
        .resource_mut::<Assets<DataArray>>()
        .add(array);
    let meta = BufferMeta {
        name: spec.id.to_string(),
        dtype,
        shape,
    };
    app.world_mut()
        .resource_mut::<DataStore>()
        .insert(id, meta, handle);
    Some(id)
}

/// Spawns an object with one actor of `kind` drawn under it, bound to arrays
/// that fit every input the kind declares.
///
/// Built through the same parameter map a client's call produces: defaults for
/// the settings, a handle for each input. Nothing here inserts a style component
/// — `apply_actor_params` derives it, which is one more thing being tested.
fn spawn(app: &mut App, kind: &'static str) -> Entity {
    let registered = app
        .world()
        .resource::<ActorRegistry>()
        .get(kind)
        .expect("the kind came from this registry");
    let mut params = registered.defaults();
    // Copied out, so the registry borrow ends before the arrays are added.
    let inputs: Vec<ParamSpec> = registered.inputs().copied().collect();

    for (index, spec) in inputs.into_iter().enumerate() {
        if let Some(id) = conforming(&spec, index as u64, app) {
            params.insert(spec.id.to_string(), ParamValue::Data(id));
        }
    }

    let object = app
        .world_mut()
        .spawn(SceneObject {
            name: kind.to_string(),
        })
        .id();
    app.world_mut()
        .spawn((
            ActorKindId(kind),
            ActorParams(params),
            Subset::All,
            crate::scene::ColorBy::default(),
            Parents(vec![object]),
            Shown(true),
            Visibility::Hidden,
        ))
        .id()
}

/// Builds the backend and runs a fresh app for each kind it registered.
///
/// One app per kind rather than one holding all of them, so a failure names the
/// kind that caused it instead of whichever ran first.
fn stand_up(backend: Backend) {
    let kinds: Vec<&'static str> = headless(backend)
        .world()
        .resource::<ActorRegistry>()
        .iter()
        .map(|kind| kind.id)
        .collect();
    assert!(
        !kinds.is_empty(),
        "the {} backend registered no kinds at all",
        backend.name()
    );

    for kind in kinds {
        // Printed before the frames rather than asserted after them: the way
        // this test fails is a panic inside the schedule, which never reaches an
        // assertion. Test output is captured unless the test fails, so this
        // costs nothing and names the kind when it matters.
        println!("{}: standing up {kind}", backend.name());

        let mut app = headless(backend);
        let actor = spawn(&mut app, kind);

        // Three frames, because the states differ. The first spawns the actor's
        // placements and draws it; the second copies what it drew onto them and
        // is the first frame anything is settled; the third runs with nothing
        // dirty, which is the state the app spends its life in.
        for _ in 0..3 {
            app.update();
        }

        assert!(
            app.world().get_entity(actor).is_ok(),
            "{}: the {kind} actor did not survive three frames",
            backend.name()
        );
        let dirty = app
            .world()
            .get::<super::Dirty>(actor)
            .copied()
            .unwrap_or_default();
        assert!(
            !dirty.any(),
            "{}: {kind} was still {dirty:?} after three frames",
            backend.name()
        );
    }
}

#[test]
fn the_default_backend_stands_up_with_every_kind_it_registers() {
    stand_up(Backend::Default);
}

#[test]
fn the_rt_backend_stands_up_with_every_kind_it_registers() {
    stand_up(Backend::Rt);
}
