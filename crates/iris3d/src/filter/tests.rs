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

use crate::scene::registry::apply_actor_params;
use iris3d_data::array::BufferMeta;
use iris3d_model::ParamValue;

/// A filter that doubles whatever it is given.
///
/// Enough to tell "the run happened" from "the run happened with these inputs",
/// which a filter producing a constant could not.
const DOUBLE: &[ParamSpec] = &[ParamSpec {
    id: "values",
    label: "values",
    kind: iris3d_model::ParamKind::Array {
        dtypes: &[Dtype::Float32],
        shape: &[0],
        required: true,
        structural: true,
    },
}];

const DOUBLED: &[OutputSpec] = &[OutputSpec {
    id: "doubled",
    label: "doubled",
    kind: super::OutputKind::Array {
        dtype: Some(Dtype::Float32),
        shape: &[0],
    },
    provenance: super::Provenance::Identity("values"),
}];

fn double(request: &Request) -> Outcome {
    let mut products = Products::new();
    let Some(values) = request.input("values") else {
        return Outcome::refused("has nothing bound to \"values\"");
    };
    let bytes: Vec<u8> = values
        .to_f32()
        .iter()
        .flat_map(|v| (v * 2.0).to_le_bytes())
        .collect();
    products.insert(
        "doubled",
        DataArray::numeric(Dtype::Float32, values.shape.clone(), bytes).into(),
    );
    products.into()
}

/// A filter that refuses an empty array and doubles anything else.
///
/// The point is that it can be made to fail and then to recover by rewriting
/// what it is bound to, which is what a test of [`FilterProblem`] needs: the
/// component has to appear *and* go away again.
fn strict(request: &Request) -> Outcome {
    let Some(values) = request.input("values") else {
        return Outcome::refused("has nothing bound to \"values\"");
    };
    let values = values.to_f32();
    if values.is_empty() {
        return Outcome::refused("was given an empty array");
    }
    let mut products = Products::new();
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| (v * 2.0).to_le_bytes())
        .collect();
    products.insert(
        "doubled",
        DataArray::numeric(Dtype::Float32, vec![values.len() as u64], bytes).into(),
    );
    products.into()
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()));
    app.init_asset::<DataArray>()
        // A geometry output is a `Mesh`, so `collect` writes into this even
        // where no kind registered here produces one.
        .init_asset::<Mesh>()
        .init_resource::<DataStore>()
        .init_resource::<crate::scene::registry::ActorRegistry>()
        // What `apply_filter_commands` writes to. Supplied by `ScenePlugin` and
        // `CounterPlugin` in the real app; this harness runs neither.
        .init_resource::<iris3d_core::counter::GlobalIDCounter>()
        .init_resource::<iris3d_core::redraw::KeepAwake>()
        .add_systems(Update, apply_actor_params);
    app.add_plugins(FilterPlugin);
    app.world_mut()
        .resource_mut::<FilterRegistry>()
        .register(FilterKind {
            id: "double",
            label: "double",
            params: DOUBLE,
            outputs: DOUBLED,
            run: Some(double),
        });
    app.world_mut()
        .resource_mut::<FilterRegistry>()
        .register(FilterKind {
            id: "strict",
            label: "strict",
            params: DOUBLE,
            outputs: DOUBLED,
            run: Some(strict),
        });
    app
}

/// What this filter is complaining of, if anything.
fn problem(app: &App, filter: Entity) -> Option<String> {
    app.world()
        .entity(filter)
        .get::<FilterProblem>()
        .map(|problem| problem.0.clone())
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
        let mut world = app
            .world_mut()
            .query::<(Option<&Stale>, Option<&Running>)>();
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
        .get(&store.array(id).expect("a held array").handle)
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
    assert!(app.world().resource::<DataStore>().array(100).is_some());
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
        .array(100)
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
        .array(100)
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
        .array(0)
        .expect("held")
        .handle
        .clone();
    let bytes: Vec<u8> = [10.0f32, 20.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
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

/// A run that fails says so, and says it where the interface and a client can
/// both read it.
///
/// Before this, the only way to report failure was to produce nothing — and
/// producing nothing leaves the previous contents standing, so a broken filter
/// and a filter that had not run yet were the same thing from outside.
#[test]
fn a_failed_run_reports_why() {
    let mut app = app();
    upload(&mut app, 0, &[]);
    let filter = spawn(&mut app, "strict", 100, 0);

    settle(&mut app);

    assert_eq!(
        problem(&app, filter).as_deref(),
        Some("was given an empty array"),
    );
}

/// And stops saying so once it works.
///
/// The half that is easy to leave out, and the half that matters more: a
/// complaint that outlives its cause sends someone hunting a fault that has
/// already been fixed.
#[test]
fn a_problem_clears_when_the_next_run_succeeds() {
    let mut app = app();
    upload(&mut app, 0, &[]);
    let filter = spawn(&mut app, "strict", 100, 0);
    settle(&mut app);
    assert!(problem(&app, filter).is_some(), "should have refused");

    // Rewrite the bound array to something usable. The `AssetEvent::Modified`
    // this raises is what marks the filter stale again.
    let source = app
        .world()
        .resource::<DataStore>()
        .array(0)
        .expect("held")
        .handle
        .clone();
    let bytes: Vec<u8> = [3.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    *app.world_mut()
        .resource_mut::<Assets<DataArray>>()
        .get_mut(&source)
        .expect("the asset") = DataArray::numeric(Dtype::Float32, vec![1], bytes);

    settle(&mut app);

    assert_eq!(problem(&app, filter), None, "the complaint should be gone");
    assert_eq!(held(&app, 100), vec![6.0]);
}

/// A refusal leaves the previous output alone rather than blanking it.
///
/// Deliberate: the last good answer is more useful on screen than nothing while
/// the reason it is stale is stated beside it. Blanking would also make a
/// downstream filter re-run against an empty array and fail in turn, so one
/// fault would cascade into a row of unrelated complaints.
#[test]
fn a_refusal_keeps_the_last_good_output() {
    let mut app = app();
    upload(&mut app, 0, &[4.0]);
    let filter = spawn(&mut app, "strict", 100, 0);
    settle(&mut app);
    assert_eq!(held(&app, 100), vec![8.0]);

    let source = app
        .world()
        .resource::<DataStore>()
        .array(0)
        .expect("held")
        .handle
        .clone();
    *app.world_mut()
        .resource_mut::<Assets<DataArray>>()
        .get_mut(&source)
        .expect("the asset") = DataArray::numeric(Dtype::Float32, vec![0], Vec::new());

    settle(&mut app);

    assert!(problem(&app, filter).is_some(), "it should be complaining");
    assert_eq!(held(&app, 100), vec![8.0], "the old answer still stands");
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

/// The command surface: adding, configuring, listing and removing filters.
///
/// Here rather than beside the scene's own command tests because these are
/// filter facts — handle allocation, cycle refusal, what a removal forgets — and
/// the scene no longer knows how to apply them. The harness runs both drains,
/// chained the way the real app does, because uploading an array is a scene
/// command and binding it is a filter one.
#[cfg(test)]
mod commands {
    use super::super::*;
    use crate::scene::registry::{self, ActorRegistry};
    use crate::scene::{CommandBus, DataStore, SceneCommand, SceneError};
    use bevy::transform::TransformPlugin;
    use iris3d_core::counter::GlobalIDCounter;
    use iris3d_core::redraw::KeepAwake;
    use iris3d_data::array::{BufferMeta, NamedBuffer};
    use tokio::sync::oneshot;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(TransformPlugin);
        app.add_message::<AssetEvent<DataArray>>();
        app.init_resource::<Assets<DataArray>>();
        app.add_message::<AssetEvent<Mesh>>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<DataStore>();
        app.init_resource::<GlobalIDCounter>();
        app.init_resource::<ActorRegistry>();
        app.init_resource::<FilterRegistry>();
        app.world_mut()
            .resource_mut::<ActorRegistry>()
            .served_by("test");
        app.init_resource::<KeepAwake>();
        app.init_resource::<CommandBus>();
        app.init_resource::<FilterBus>();
        // Both drains, in the order the real app chains them: the scene's first,
        // so an array uploaded this tick is in the store when a filter created
        // this tick binds it.
        app.add_systems(
            Update,
            (
                crate::scene::apply_scene_commands,
                crate::filter::wire::apply_filter_commands,
            )
                .chain(),
        );
        app
    }

    fn send<T>(
        app: &App,
        make: impl FnOnce(oneshot::Sender<T>) -> SceneCommand,
    ) -> oneshot::Receiver<T> {
        let (tx, rx) = oneshot::channel();
        app.world()
            .resource::<CommandBus>()
            .sender()
            .send(make(tx))
            .expect("the scene is draining");
        rx
    }

    fn send_filter<T>(
        app: &App,
        make: impl FnOnce(oneshot::Sender<T>) -> FilterCommand,
    ) -> oneshot::Receiver<T> {
        let (tx, rx) = oneshot::channel();
        app.world()
            .resource::<FilterBus>()
            .sender()
            .send(make(tx))
            .expect("the filter graph is draining");
        rx
    }

    fn array(name: &str, bytes: usize) -> NamedBuffer {
        NamedBuffer {
            meta: BufferMeta {
                name: name.into(),
                dtype: Dtype::Uint8,
                shape: vec![bytes as u64],
            },
            data: vec![0; bytes],
            strings: Vec::new(),
        }
    }
    /// A filter kind that reads one array and writes two, so a test can tell
    /// "an output" from "the outputs" and check declaration order.
    fn passthrough(app: &mut App) {
        const PARAMS: &[iris3d_model::ParamSpec] = &[iris3d_model::ParamSpec {
            id: "values",
            label: "values",
            kind: iris3d_model::ParamKind::Array {
                dtypes: &[],
                shape: &[],
                required: true,
                structural: true,
            },
        }];
        const OUTPUTS: &[super::super::OutputSpec] = &[
            super::super::OutputSpec {
                id: "first",
                label: "first",
                kind: super::super::OutputKind::Array {
                    dtype: Some(Dtype::Uint8),
                    shape: &[0],
                },
                provenance: super::super::Provenance::Opaque,
            },
            super::super::OutputSpec {
                id: "second",
                label: "second",
                kind: super::super::OutputKind::Array {
                    dtype: Some(Dtype::Uint8),
                    shape: &[0],
                },
                provenance: super::super::Provenance::Opaque,
            },
        ];

        app.world_mut()
            .resource_mut::<super::super::FilterRegistry>()
            .register(super::super::FilterKind {
                id: "passthrough",
                label: "passthrough",
                params: PARAMS,
                outputs: OUTPUTS,
                run: Some(|_| super::super::Products::new().into()),
            });
    }

    fn add_filter(
        app: &App,
        kind: &str,
        bound: u64,
    ) -> oneshot::Receiver<Result<FilterSummary, SceneError>> {
        let mut params = iris3d_model::ParamMap::new();
        params.insert("values".to_string(), iris3d_model::ParamValue::Data(bound));
        send_filter(app, |reply| FilterCommand::Add {
            kind: kind.into(),
            params,
            reply,
        })
    }

    /// One upload, and the handle it came back as.
    fn one_array(app: &mut App) -> u64 {
        let mut uploaded = send(app, |reply| SceneCommand::UploadData {
            arrays: vec![array("values", 4)],
            reply,
        });
        app.update();
        uploaded.try_recv().expect("a reply")[0].id
    }

    /// The reply carries usable handles, so the next call can bind one. Without
    /// this every caller would have to add a filter, wait, and ask again.
    #[test]
    fn adding_a_filter_allocates_a_handle_per_declared_output() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let names: Vec<&str> = summary
            .outputs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, vec!["first", "second"], "declaration order");

        let store = app.world().resource::<DataStore>();
        for (name, handle) in &summary.outputs {
            let held = store
                .array(*handle)
                .expect("registered before the first run");
            assert_eq!(&held.meta.name, name);
        }
    }

    /// Naming a kind this build does not have is the caller's mistake, and the
    /// reply says how to find out what does exist.
    #[test]
    fn adding_an_unregistered_filter_kind_is_refused() {
        let mut app = app();
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "no-such-filter", values);
        app.update();
        assert!(matches!(
            added.try_recv().expect("a reply"),
            Err(SceneError::UnknownFilterKind { .. })
        ));
    }

    /// A filter that fed itself would never come to rest: each run rewrites an
    /// array that marks the next one stale, forever, with the app awake
    /// throughout. Refusing the binding is much cheaper than detecting the spin.
    #[test]
    fn a_filter_cannot_be_made_to_read_its_own_output() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");
        let (_, own_output) = summary.outputs[0].clone();

        let mut params = iris3d_model::ParamMap::new();
        params.insert(
            "values".to_string(),
            iris3d_model::ParamValue::Data(own_output),
        );
        let mut set = send_filter(&app, |reply| FilterCommand::Set {
            id: summary.id,
            params,
            reply,
        });
        app.update();

        assert!(matches!(
            set.try_recv().expect("a reply"),
            Err(SceneError::FilterCycle { .. })
        ));
    }

    /// The indirect case, which is the one a caller cannot see coming: two
    /// filters that each look reasonable on their own.
    #[test]
    fn a_cycle_through_another_filter_is_refused() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut first = add_filter(&app, "passthrough", values);
        app.update();
        let first = first.try_recv().expect("a reply").expect("added");

        // The second reads the first. Fine so far.
        let mut second = add_filter(&app, "passthrough", first.outputs[0].1);
        app.update();
        let second = second.try_recv().expect("a reply").expect("added");

        // Now point the first at the second, closing the loop.
        let mut params = iris3d_model::ParamMap::new();
        params.insert(
            "values".to_string(),
            iris3d_model::ParamValue::Data(second.outputs[0].1),
        );
        let mut set = send_filter(&app, |reply| FilterCommand::Set {
            id: first.id,
            params,
            reply,
        });
        app.update();

        assert!(matches!(
            set.try_recv().expect("a reply"),
            Err(SceneError::FilterCycle { .. })
        ));
    }

    /// Releasing an array a filter writes would leave it producing into nothing,
    /// which on screen is indistinguishable from a filter that has broken.
    #[test]
    fn a_filters_output_cannot_be_released_on_its_own() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");
        let generated = summary.outputs[0].1;

        let mut released = send(&app, |reply| SceneCommand::ReleaseData {
            // The upload alongside it, to show one refusal does not lose the
            // rest of the batch.
            ids: vec![generated, values],
            reply,
        });
        app.update();

        assert_eq!(released.try_recv().expect("a reply"), vec![values]);
        assert!(
            app.world()
                .resource::<DataStore>()
                .array(generated)
                .is_some(),
            "still held, because the filter is still writing it"
        );
    }

    /// Removing the filter *is* how those handles go away.
    #[test]
    fn removing_a_filter_forgets_the_arrays_it_was_writing() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let mut removed = send_filter(&app, |reply| FilterCommand::Remove {
            id: summary.id,
            reply,
        });
        app.update();
        assert!(removed.try_recv().expect("a reply"));

        let store = app.world().resource::<DataStore>();
        for (_, handle) in &summary.outputs {
            assert!(store.array(*handle).is_none(), "released with the filter");
        }
        assert!(store.array(values).is_some(), "the upload is untouched");
    }

    /// A listing is how a client rediscovers a scene it did not build, so it has
    /// to carry the output handles as well as the settings.
    #[test]
    fn listing_filters_reports_their_outputs() {
        let mut app = app();
        passthrough(&mut app);
        let values = one_array(&mut app);

        let mut added = add_filter(&app, "passthrough", values);
        app.update();
        let summary = added.try_recv().expect("a reply").expect("added");

        let mut listed = send_filter(&app, |reply| FilterCommand::List { reply });
        app.update();
        let filters = listed.try_recv().expect("a reply");

        assert_eq!(filters.len(), 1);
        assert_eq!(filters[0].id, summary.id);
        assert_eq!(filters[0].kind, "passthrough");
        assert_eq!(filters[0].outputs, summary.outputs);
    }
}
