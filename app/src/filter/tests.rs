//! The runner, rather than any particular filter.
//!
//! What is asserted here is the machinery every filter depends on and none of
//! them implements: that a run happens at all, that its result reaches the
//! handle a consumer bound *before* it ran, that a stale answer is thrown away,
//! and that one filter reading another's output re-runs when it changes.
//!
//! A run goes to [`AsyncComputeTaskPool`], so nothing lands on the tick that
//! starts it. Every test here drives frames until it does — see [`settle`].

use super::*;

use crate::scene::data::BufferMeta;
use crate::scene::registry::{ParamValue, apply_actor_params};

/// A filter that doubles whatever it is given.
///
/// Enough to tell "the run happened" from "the run happened with these inputs",
/// which a filter producing a constant could not.
const DOUBLE: &[ParamSpec] = &[ParamSpec {
    id: "values",
    label: "values",
    kind: crate::scene::registry::ParamKind::Array {
        dtypes: &[Dtype::Float32],
        shape: &[0],
        required: true,
    },
}];

const DOUBLED: &[OutputSpec] = &[OutputSpec {
    id: "doubled",
    label: "doubled",
    dtype: Dtype::Float32,
    shape: &[0],
}];

fn double(request: &Request) -> Products {
    let mut products = Products::new();
    let Some(values) = request.input("values") else {
        return products;
    };
    let bytes: Vec<u8> = values
        .to_f32()
        .iter()
        .flat_map(|v| (v * 2.0).to_le_bytes())
        .collect();
    products.insert(
        "doubled",
        DataArray::numeric(Dtype::Float32, values.shape.clone(), bytes),
    );
    products
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()));
    app.init_asset::<DataArray>()
        .init_resource::<DataStore>()
        .init_resource::<crate::scene::registry::ActorRegistry>()
        .add_systems(Update, apply_actor_params);
    app.add_plugins(FilterPlugin);
    app.world_mut()
        .resource_mut::<FilterRegistry>()
        .register(FilterKind {
            id: "double",
            label: "double",
            params: DOUBLE,
            outputs: DOUBLED,
            run: double,
        });
    app
}

/// Puts an array in the store under `id`, as an upload would.
fn upload(app: &mut App, id: u64, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let array = DataArray::numeric(Dtype::Float32, vec![values.len() as u64], bytes);
    let handle = app
        .world_mut()
        .resource_mut::<Assets<DataArray>>()
        .add(array);
    app.world_mut().resource_mut::<DataStore>().insert(
        id,
        BufferMeta {
            name: "values".into(),
            dtype: Dtype::Float32,
            shape: vec![values.len() as u64],
        },
        handle,
    );
}

/// Spawns a filter with its outputs allocated, which is what the scene does on
/// `AddFilter`.
fn spawn(app: &mut App, kind: &'static str, first_output_handle: u64, bound: u64) -> Entity {
    let outputs: Vec<&'static str> = app
        .world()
        .resource::<FilterRegistry>()
        .get(kind)
        .expect("registered")
        .outputs
        .iter()
        .map(|spec| spec.id)
        .collect();

    let mut allocated = HashMap::new();
    for (offset, output) in outputs.into_iter().enumerate() {
        let id = first_output_handle + offset as u64;
        let handle = app
            .world_mut()
            .resource_mut::<Assets<DataArray>>()
            .add(DataArray::numeric(Dtype::Float32, vec![0], Vec::new()));
        app.world_mut().resource_mut::<DataStore>().insert(
            id,
            BufferMeta {
                name: output.to_string(),
                dtype: Dtype::Float32,
                shape: vec![0],
            },
            handle,
        );
        allocated.insert(output, id);
    }

    let mut params = ParamMap::new();
    params.insert("values".to_string(), ParamValue::Data(bound));
    app.world_mut()
        .spawn((
            FilterKindId(kind),
            FilterParams(params),
            Generation::default(),
            Outputs(allocated),
        ))
        .id()
}

/// Runs frames until nothing is stale or in flight, and stays that way.
///
/// Quiet for **one** frame is not settled. A finished run rewrites an asset, and
/// `AssetEvent::Modified` is not delivered until the next frame — so on the tick
/// a filter finishes, everything downstream of it is momentarily idle and has
/// not yet been told. Requiring several consecutive quiet frames is what lets a
/// chain of any length finish.
///
/// Bounded rather than looping forever, so a filter that never settles fails as
/// a test rather than hanging one.
fn settle(app: &mut App) {
    let mut quiet = 0;
    for _ in 0..200 {
        app.update();
        let mut world = app.world_mut().query::<(Option<&Stale>, Option<&Running>)>();
        let busy = world
            .iter(app.world())
            .any(|(stale, running)| stale.is_some() || running.is_some());
        quiet = match busy {
            true => 0,
            false => quiet + 1,
        };
        if quiet >= 4 {
            return;
        }
    }
    panic!("a filter never settled");
}

/// What is held under a handle right now.
fn held(app: &App, id: u64) -> Vec<f32> {
    let store = app.world().resource::<DataStore>();
    let arrays = app.world().resource::<Assets<DataArray>>();
    arrays
        .get(&store.get(id).expect("a held array").handle)
        .expect("the asset")
        .to_f32()
}

#[test]
fn a_filter_runs_and_writes_its_declared_output() {
    let mut app = app();
    upload(&mut app, 0, &[1.0, 2.0, 3.0]);
    spawn(&mut app, "double", 100, 0);

    settle(&mut app);

    assert_eq!(held(&app, 100), vec![2.0, 4.0, 6.0]);
}

/// The handle exists from the moment the filter does, so a client can bind an
/// output in the same breath as creating the filter that fills it. Without this
/// every caller would have to wait a frame and ask again.
#[test]
fn an_output_handle_is_bindable_before_the_first_run() {
    let mut app = app();
    upload(&mut app, 0, &[1.0]);
    spawn(&mut app, "double", 100, 0);

    // No frames at all: the handle is already in the store.
    assert!(app.world().resource::<DataStore>().get(100).is_some());
    assert_eq!(held(&app, 100), Vec::<f32>::new(), "empty until it has run");
}

/// The point of rewriting the asset rather than replacing the handle. A
/// consumer binds once and keeps that binding for the filter's whole life.
#[test]
fn re_running_keeps_the_same_handle() {
    let mut app = app();
    upload(&mut app, 0, &[1.0]);
    let filter = spawn(&mut app, "double", 100, 0);
    settle(&mut app);

    let before = app
        .world()
        .resource::<DataStore>()
        .get(100)
        .expect("held")
        .handle
        .id();

    // A parameter change is one of the three reasons to re-run.
    app.world_mut()
        .entity_mut(filter)
        .get_mut::<FilterParams>()
        .expect("params")
        .0
        .insert("values".to_string(), ParamValue::Data(0));
    settle(&mut app);

    let after = app
        .world()
        .resource::<DataStore>()
        .get(100)
        .expect("held")
        .handle
        .id();
    assert_eq!(before, after, "the asset was rewritten, not replaced");
}

/// Rewriting the bytes behind a bound handle is how a filter chain propagates,
/// and it has to work without the binding itself changing.
#[test]
fn rewriting_a_bound_array_re_runs_the_filter() {
    let mut app = app();
    upload(&mut app, 0, &[1.0, 2.0]);
    spawn(&mut app, "double", 100, 0);
    settle(&mut app);
    assert_eq!(held(&app, 100), vec![2.0, 4.0]);

    let source = app
        .world()
        .resource::<DataStore>()
        .get(0)
        .expect("held")
        .handle
        .clone();
    let bytes: Vec<u8> = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    *app.world_mut()
        .resource_mut::<Assets<DataArray>>()
        .get_mut(&source)
        .expect("the asset") = DataArray::numeric(Dtype::Float32, vec![2], bytes);

    settle(&mut app);
    assert_eq!(held(&app, 100), vec![20.0, 40.0]);
}

/// Two filters in a line, the second reading the first's output.
///
/// This is the whole reason outputs are ordinary arrays in the store: nothing
/// walks a graph, and the second filter cannot tell an upstream filter from an
/// upload.
#[test]
fn one_filter_reads_another() {
    let mut app = app();
    upload(&mut app, 0, &[1.0, 2.0]);
    spawn(&mut app, "double", 100, 0);
    spawn(&mut app, "double", 200, 100);

    settle(&mut app);

    assert_eq!(held(&app, 100), vec![2.0, 4.0]);
    assert_eq!(held(&app, 200), vec![4.0, 8.0], "doubled twice");
}

/// A run describes the inputs it started with. If those have moved on, its
/// answer is wrong — and writing it would leave whichever task finished last on
/// screen rather than whichever is current.
#[test]
fn a_run_that_went_stale_is_discarded() {
    let mut app = app();
    upload(&mut app, 0, &[1.0]);
    upload(&mut app, 1, &[5.0]);
    let filter = spawn(&mut app, "double", 100, 0);

    // Started against handle 0, then rebound before it can land.
    app.update();
    app.world_mut()
        .entity_mut(filter)
        .get_mut::<FilterParams>()
        .expect("params")
        .0
        .insert("values".to_string(), ParamValue::Data(1));

    settle(&mut app);

    assert_eq!(
        held(&app, 100),
        vec![10.0],
        "the answer for the inputs it ended with, not the ones it began with"
    );
}

/// A filter's `run` never touches the world, so it cannot be the thing that
/// keeps the app alive or brings it down. An unregistered kind is the ordinary
/// way a stale scene refers to something this build no longer has.
#[test]
fn an_unregistered_kind_is_survivable() {
    let mut app = app();
    app.world_mut().spawn((
        FilterKindId("nothing-runs-this"),
        FilterParams::default(),
        Generation::default(),
        Outputs::default(),
    ));

    // It never leaves `Stale`, because nothing can run it — so `settle` would
    // rightly fail. Three frames is enough to show nothing panics.
    for _ in 0..3 {
        app.update();
    }
}
