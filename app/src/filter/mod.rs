//! Arrays in, arrays out. Nothing here draws.
//!
//! A **filter** reads arrays and parameters and writes arrays. An **actor**
//! reads arrays and draws them. That line is the whole point of this module,
//! and it exists because actor kinds used to sit on both sides of it.
//!
//! # What went wrong without it
//!
//! [`draw::cartoon`](crate::draw::cartoon) builds the triangles of a ribbon and
//! is careful to hand back a `Ribbon` rather than a `Mesh`, so that any pathway
//! could map it onto GPU data its own way. But its only caller was one actor's
//! draw system. The ribbon lived for one frame, inside one actor, and nothing
//! else could see it — so drawing the same ribbon as an absorbing medium meant
//! giving `cartoon` a `mode` parameter that duplicated the difference between
//! the `mesh` and `solid` actor kinds.
//!
//! That is the combinatorial shape: every kind that *generates* geometry grows a
//! mode for every way of *displaying* it, and two ways of displaying one ribbon
//! build it twice. Filters make it `N + M` instead of `N * M`, and let two
//! actors share one generated result.
//!
//! # Filters are above the backends
//!
//! A filter produces arrays, not GPU data, so it knows nothing about pipelines
//! and lives outside [`draw`](crate::draw) entirely. The same `contour` output
//! feeds whatever can draw triangles.
//!
//! # How a filter takes part
//!
//! It is an entity, carrying the same components an actor does — a kind id,
//! a [`ParamMap`], and [`Bindings`] derived from it — plus [`Outputs`], which
//! maps each declared output to the handle it writes.
//!
//! Those handles are allocated **when the filter is created** and are stable for
//! its life, so a client can bind an output before the first run has produced
//! anything. Each starts as an empty array in [`DataStore`]. A run rewrites the
//! asset in place rather than replacing the handle, which is what makes
//! [`draw::mark_dirty`](crate::draw) re-dirty every actor bound to it with no
//! new code — it already watches `AssetEvent::Modified`.
//!
//! Chaining falls out of the same fact. Filter A rewrites its output, the asset
//! event marks filter B stale, B runs and rewrites its own, and the actor at the
//! end redraws. Nothing walks a graph, and no filter knows its consumers.
//!
//! The price of that is **one frame per link**: `AssetEvent::Modified` is not
//! delivered until the frame after the write, so a two-filter chain reaches the
//! screen a frame later than a one-filter chain. Worth knowing and not worth
//! removing — removing it means walking the graph in dependency order, which is
//! the coupling this shape exists to avoid, to save a frame on a chain that has
//! already spent several running.
//!
//! # Off the main thread
//!
//! A run happens on [`AsyncComputeTaskPool`], because the ones worth having are
//! not frame-sized: extracting a surface from a 256³ grid is not something to do
//! between two frames. The cost of that is a copy — a task cannot borrow from
//! the world, so it takes owned input arrays. See [`Request`].

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};

use crate::scene::data::{BufferMeta, Dtype};
use crate::scene::registry::{Bindings, ParamMap, ParamSpec, data as bound_handle};
use crate::scene::{DataArray, DataStore};

pub(crate) mod colormap;
mod wire;

pub use wire::{FilterKindSummary, FilterSummary, Filters, Graph};
pub(crate) use wire::{add, list, list_kinds, remove, set};

/// One array a filter writes.
///
/// The mirror of [`ParamKind::Array`](crate::scene::registry::ParamKind::Array)
/// on the way in, and declared for the same reason: a client is told that
/// `colormap` produces `float32 [n, 3]` rather than having to run it and look.
#[derive(Debug, Clone, Copy)]
pub struct OutputSpec {
    /// Stable identifier, used as the key in [`Outputs`] and on the wire.
    pub id: &'static str,
    pub label: &'static str,
    pub dtype: Dtype,
    /// Shape, where `0` is an axis decided at run time — `[0, 3]` for a colour
    /// per element. A contour's vertex count is not knowable before it runs, so
    /// this describes the *form* of the output and never its size.
    pub shape: &'static [u64],
}

/// Everything one run gets, owned.
///
/// Owned rather than borrowed because the run happens on a worker thread, which
/// cannot hold a reference into the world. The cost is a copy of every bound
/// input per run, which is real — a 256³ float grid is 64 MB — and is the price
/// of not blocking the frame. A GPU filter will not pay it, because its inputs
/// are already on the device and never come back.
pub struct Request {
    pub params: ParamMap,
    /// The arrays bound to this filter's inputs, by input id. An optional input
    /// nobody bound is simply absent.
    pub inputs: HashMap<&'static str, DataArray>,
}

impl Request {
    /// A bound input, or `None` if nothing was bound to it.
    pub fn input(&self, id: &str) -> Option<&DataArray> {
        self.inputs.get(id)
    }
}

/// What a run produced, keyed by [`OutputSpec::id`].
///
/// A run that cannot produce something — degenerate input, an unbound optional
/// it turned out to need — leaves it out rather than inventing an array. The
/// previous contents then stand, which is the honest outcome: nothing was
/// learned, so nothing changes.
pub type Products = HashMap<&'static str, DataArray>;

/// A way of deriving data, as declared by whatever implements it.
///
/// The shape deliberately mirrors
/// [`ActorKind`](crate::scene::registry::ActorKind): same `ParamSpec`
/// declarations, same defaulting, same normalisation. A filter and an actor are
/// configured identically and differ only in what comes out of them.
pub struct FilterKind {
    /// Stable identifier — `"colormap"`, `"cartoon"`, `"contour"`. Goes over
    /// the wire.
    pub id: &'static str,
    pub label: &'static str,
    pub params: &'static [ParamSpec],
    pub outputs: &'static [OutputSpec],
    /// Does the work. Runs on a worker thread, so it may take as long as it
    /// needs and must not touch the world.
    pub run: fn(&Request) -> Products,
}

impl FilterKind {
    /// Every input this kind reads an array from, required or not.
    pub fn inputs(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params
            .iter()
            .filter(|spec| matches!(spec.kind, crate::scene::registry::ParamKind::Array { .. }))
    }

    /// A complete, in-range parameter map built from whatever was supplied. See
    /// [`ActorKind::normalise`](crate::scene::registry::ActorKind::normalise).
    pub fn normalise(&self, given: &ParamMap) -> ParamMap {
        self.params
            .iter()
            .filter_map(|spec| {
                let value = given
                    .get(spec.id)
                    .cloned()
                    .and_then(|value| spec.kind.sanitise(value))
                    .or_else(|| spec.kind.default_value())?;
                Some((spec.id.to_string(), value))
            })
            .collect()
    }
}

/// Every filter kind this build can run.
///
/// Unlike [`ActorRegistry`](crate::scene::registry::ActorRegistry) this is not
/// filled by a backend: a filter derives data and has no pipeline, so the same
/// set exists whatever draws afterwards.
#[derive(Resource, Default)]
pub struct FilterRegistry {
    kinds: Vec<FilterKind>,
}

impl FilterRegistry {
    pub fn register(&mut self, kind: FilterKind) {
        if let Some(existing) = self.kinds.iter_mut().find(|existing| existing.id == kind.id) {
            warn!("filter: kind \"{}\" re-registered", kind.id);
            *existing = kind;
            return;
        }
        self.kinds.push(kind);
    }

    pub fn get(&self, id: &str) -> Option<&FilterKind> {
        self.kinds.iter().find(|kind| kind.id == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &FilterKind> {
        self.kinds.iter()
    }
}

/// Which registered kind a filter is.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterKindId(pub &'static str);

/// A filter's parameters — the authoritative copy, as
/// [`ActorParams`](crate::scene::registry::ActorParams) is for an actor.
#[derive(Component, Debug, Clone, Default, PartialEq)]
pub struct FilterParams(pub ParamMap);

/// Where this filter's declared outputs live, by output id.
///
/// Allocated once, at creation, and never reallocated: a consumer binds a handle
/// and keeps it, however many times the contents are rewritten.
#[derive(Component, Debug, Clone, Default)]
pub struct Outputs(pub HashMap<&'static str, u64>);

impl Outputs {
    pub fn get(&self, output: &str) -> Option<u64> {
        self.0.get(output).copied()
    }
}

/// This filter's results no longer follow from its inputs.
///
/// Not graded the way [`Dirty`](crate::draw::Dirty) is. An actor can repaint
/// without re-tessellating, so it is worth saying which half went stale; a
/// filter has one product and one way to get it.
#[derive(Component, Debug, Default)]
pub struct Stale;

/// How many times this filter has gone stale.
///
/// A run carries the generation it started under. If the filter has gone stale
/// again since, the result describes inputs that no longer apply and is thrown
/// away rather than written — otherwise dragging a slider would leave whichever
/// task happened to finish last on screen.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Generation(pub u64);

/// A run in flight.
#[derive(Component)]
pub struct Running {
    task: Task<Products>,
    started_at: Generation,
}

/// A filter whose settings or bindings moved this tick.
///
/// Insertion counts as a change, so a filter spawned this tick is caught here
/// rather than needing an `Added` filter of its own.
type Reconfigured = (
    With<FilterKindId>,
    Or<(Changed<FilterParams>, Changed<Bindings>)>,
);

/// A filter waiting to be started: everything [`start`] needs to build a
/// [`Request`] from it.
type Startable<'a> = (
    Entity,
    &'a FilterKindId,
    &'a FilterParams,
    &'a Bindings,
    &'a Generation,
);

/// Ordering label for everything in this module, so a backend can put its own
/// work after it.
///
/// Before [`Invalidate`](crate::draw) rather than inside it: a filter finishing
/// rewrites an array, and the actors bound to that array have to be marked
/// out of date on the same tick, not the next one.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Filter;

pub struct FilterPlugin;

impl Plugin for FilterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FilterRegistry>();
        colormap::register(&mut app.world_mut().resource_mut::<FilterRegistry>());

        app.configure_sets(
            Update,
            Filter.after(crate::scene::registry::apply_actor_params),
        )
        .add_systems(
            Update,
            (apply_filter_params, mark_stale, start, collect)
                .chain()
                .in_set(Filter),
        );
    }
}

/// Regenerates [`Bindings`] from the parameters, exactly as
/// [`apply_actor_params`](crate::scene::registry::apply_actor_params) does.
///
/// A filter has no typed style component to write — nothing reads its settings
/// but its own `run`, which is handed the map — so this is the binding half
/// alone.
fn apply_filter_params(
    mut commands: Commands,
    registry: Res<FilterRegistry>,
    changed: Query<
        (Entity, &FilterKindId, &FilterParams, Option<&Bindings>),
        Changed<FilterParams>,
    >,
) {
    for (entity, kind, params, bound) in &changed {
        let Some(registered) = registry.get(kind.0) else {
            warn!("filter: no kind \"{}\" is registered", kind.0);
            continue;
        };

        let wanted = Bindings(
            registered
                .inputs()
                .filter_map(|spec| Some((spec.id, bound_handle(&params.0, spec.id)?)))
                .collect(),
        );
        // Only when it differs, so a slider drag does not look like a rebind.
        if bound != Some(&wanted) {
            commands.entity(entity).insert(wanted);
        }
    }
}

/// Decides which filters have to run again.
///
/// The reasons are the same three an actor has, and the last one is what makes
/// a chain work: an array a filter reads was rewritten, which is exactly what
/// happens when the filter upstream of it finishes.
fn mark_stale(
    mut commands: Commands,
    mut generations: Query<&mut Generation>,
    reconfigured: Query<Entity, Reconfigured>,
    mut array_events: MessageReader<AssetEvent<DataArray>>,
    store: Res<DataStore>,
    bindings: Query<(Entity, &Bindings), With<FilterKindId>>,
) {
    let mut stale: Vec<Entity> = reconfigured.iter().collect();

    let modified: Vec<_> = array_events
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    if !modified.is_empty() {
        for (entity, bound) in &bindings {
            let touched = bound.0.values().any(|id| {
                store
                    .get(*id)
                    .is_some_and(|held| modified.contains(&held.handle.id()))
            });
            if touched {
                stale.push(entity);
            }
        }
    }

    stale.sort_unstable();
    stale.dedup();
    for entity in stale {
        // Bumped whether or not a run is in flight: that is what tells a
        // finishing task its answer is out of date.
        if let Ok(mut generation) = generations.get_mut(entity) {
            generation.0 += 1;
        }
        commands.entity(entity).insert(Stale);
    }
}

/// Gathers a stale filter's inputs and puts the work on the task pool.
///
/// A filter already running is left alone rather than being restarted. The
/// generation it carries has already moved, so its result will be discarded when
/// it lands and this one starts then — which costs one stale run and keeps the
/// number in flight at one per filter however fast a slider moves.
fn start(
    mut commands: Commands,
    registry: Res<FilterRegistry>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    stale: Query<Startable, (With<Stale>, Without<Running>)>,
) {
    let pool = AsyncComputeTaskPool::get();
    for (entity, kind, params, bound, generation) in &stale {
        let Some(registered) = registry.get(kind.0) else {
            continue;
        };

        let mut inputs = HashMap::new();
        for spec in registered.inputs() {
            let Some(handle) = bound.get(spec.id) else {
                continue;
            };
            let Some(array) = store.get(handle).and_then(|held| arrays.get(&held.handle)) else {
                continue;
            };
            inputs.insert(spec.id, array.clone());
        }

        let request = Request {
            params: params.0.clone(),
            inputs,
        };
        let run = registered.run;
        commands
            .entity(entity)
            .remove::<Stale>()
            .insert(Running {
                task: pool.spawn(async move { run(&request) }),
                started_at: *generation,
            });
    }
}

/// Writes a finished run into the output arrays.
///
/// Rewrites the existing assets rather than replacing the handles, which is what
/// keeps a consumer's binding valid and what raises the `AssetEvent::Modified`
/// everything downstream is watching.
fn collect(
    mut commands: Commands,
    mut arrays: ResMut<Assets<DataArray>>,
    mut store: ResMut<DataStore>,
    mut running: Query<(Entity, &mut Running, &Generation, &Outputs, &FilterKindId)>,
) {
    for (entity, mut run, generation, outputs, kind) in &mut running {
        let Some(products) = block_on(future::poll_once(&mut run.task)) else {
            continue;
        };
        commands.entity(entity).remove::<Running>();

        // Stale before it finished. Marking it again would be wrong — it is
        // already marked, which is why the generation moved — so this only
        // drops the answer.
        if run.started_at != *generation {
            continue;
        }

        for (output, produced) in products {
            let Some(handle) = outputs.get(output) else {
                warn!("filter: {} produced undeclared output \"{output}\"", kind.0);
                continue;
            };
            let Some(held) = store.get(handle) else {
                continue;
            };
            let meta = BufferMeta {
                name: output.to_string(),
                dtype: produced.dtype,
                shape: produced.shape.clone(),
            };
            let asset = held.handle.clone();
            let Some(mut existing) = arrays.get_mut(&asset) else {
                continue;
            };
            *existing = produced;
            // The meta as well as the bytes: an output's length is decided by
            // the run, so a consumer asking what shape this handle is has to be
            // told the new answer rather than the one it was created with.
            store.insert(handle, meta, asset);
        }
    }
}

#[cfg(test)]
mod tests;
