//! Arrays in, arrays out. Nothing here draws.
//!
//! A **filter** reads arrays and parameters and writes arrays. An **actor**
//! reads arrays and draws them. Nothing straddles that line, and keeping it
//! that way is what turns `N` ways of generating geometry times `M` ways of
//! displaying it into `N + M`: one generated ribbon can be drawn as a lit
//! surface and as an absorbing medium at once.
//!
//! A filter produces arrays, not GPU data, so it knows nothing about pipelines
//! and lives outside [`draw`](crate::draw) entirely.
//!
//! # How a filter takes part
//!
//! It is an entity, carrying the same components an actor does — a kind id,
//! a [`ParamMap`], and [`Bindings`] derived from it — plus [`Outputs`], which
//! maps each declared output to the handle it writes. Those handles are
//! allocated **when the filter is created** and are stable for its life, so a
//! client can bind an output before the first run has produced anything.
//!
//! A run rewrites its output asset in place rather than replacing the handle.
//! That raises `AssetEvent::Modified`, which is already what
//! [`draw::mark_dirty`](crate::draw) watches — so consumers re-dirty with no
//! new code, and chaining falls out of the same fact. Nothing walks a graph,
//! and no filter knows its consumers. The price is one frame per link.
//!
//! A run happens on [`AsyncComputeTaskPool`], off the main thread, so it takes
//! owned copies of its inputs. See [`Request`].
//!
//! The full argument — why the split exists, what the one-frame cost buys, and
//! why an [`Outcome`] reports a problem rather than an empty output — is in
//! `docs/design/filters.md`.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};

use crate::scene::data::{BufferMeta, Dtype};
use crate::scene::registry::{Bindings, ParamMap, ParamSpec, data as bound_handle};
use crate::scene::{DataArray, DataStore};

pub(crate) mod cartoon;
pub(crate) mod colormap;
pub(crate) mod contour;
pub(crate) mod geometry;
pub(crate) mod index;
pub(crate) mod maths;
pub(crate) mod provenance;
pub(crate) mod select;
pub(crate) mod source;
mod wire;

pub use wire::{FilterBus, FilterCommand, FilterKindSummary, FilterSummary};

/// One thing a filter writes.
///
/// The mirror of [`ParamKind`](crate::scene::registry::ParamKind) on the way in,
/// and declared for the same reason: a client is told that `colormap` produces
/// `float32 [n, 3]` rather than having to run it and look.
#[derive(Debug, Clone, Copy)]
pub struct OutputSpec {
    /// Stable identifier, used as the key in [`Outputs`] and on the wire.
    pub id: &'static str,
    pub label: &'static str,
    pub kind: OutputKind,
    /// Where this output's elements came from. See [`Provenance`].
    pub provenance: Provenance,
}

/// How an output's elements correspond to an input's.
///
/// # Why this has to be declared
///
/// Subsetting is filters now, so an actor draws arrays that were narrowed,
/// gathered and renumbered several steps upstream. A click therefore lands on a
/// *drawn* element — vertex 40 122 of a ribbon — and the thing a client cares
/// about is several filters back: which atom, which residue, which chain.
///
/// Walking that back needs each filter to say how its output lines up with its
/// input. It needs no new computation, because **the correspondence is almost
/// always an array the filter already emits**: `cartoon` has emitted
/// `residue_index` per vertex since the day it was written, for colouring, and
/// that is exactly the vertex → residue map. `gather` is handed its own inverse.
/// `subset` *is* a correspondence. So this is a declaration in the style of
/// [`OutputKind`], not a feature.
///
/// It is declared beside the output rather than in a table somewhere central so
/// that adding a filter means answering the question, rather than forgetting to.
#[derive(Debug, Clone, Copy)]
pub enum Provenance {
    /// Element *i* out came from element *i* of this input.
    Identity(&'static str),
    /// Element *i* out came from element `via[i]` of `of`, where `via` is one of
    /// this filter's own outputs.
    Map {
        /// The output holding the indices, by [`OutputSpec::id`].
        via: &'static str,
        /// The input they index into, by [`ParamSpec::id`].
        of: &'static str,
    },
    /// No correspondence. The honest answer for an output whose elements are not
    /// derived from any one input's — a mask over a dictionary, or geometry
    /// whose vertices belong to cells rather than to elements.
    Opaque,
}

/// What sort of thing an output is.
///
/// The same two things a handle can name — see [`DataStore`] — because a
/// filter's outputs are ordinary handles and a consumer cannot tell one from an
/// upload.
#[derive(Debug, Clone, Copy)]
pub enum OutputKind {
    Array {
        /// Element type, or `None` when the run decides it.
        ///
        /// `None` is the same escape hatch `0` already is in `shape`, one level
        /// across: some filters cannot know their output's *type* in advance any
        /// more than they can know its length. `gather` is the case that forced
        /// it — it hands back elements of whatever it was given, and an
        /// `elements` array gathered into `ball-and-stick` has to still be
        /// `Uint8` or the binding is refused.
        ///
        /// Declaring a concrete type where one is known is still worth doing: it
        /// is what lets an output be bound *before* the first run, which is the
        /// whole reason handles are minted at creation.
        dtype: Option<Dtype>,
        /// Shape, where `0` is an axis decided at run time — `[0, 3]` for a
        /// colour per element. A contour's vertex count is not knowable before
        /// it runs, so this describes the *form* of the output and never its
        /// size.
        shape: &'static [u64],
    },
    /// One mesh, which every consumer references rather than copies.
    ///
    /// Nothing further to declare: what a mesh carries is decided by the run,
    /// and a consumer reads it off
    /// [`GeometryMeta`](crate::scene::data::GeometryMeta) afterwards rather than
    /// being promised it in advance.
    Geometry,
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

/// One finished output.
///
/// A `Mesh` is built on the worker thread like everything else here. It is an
/// engine type rather than a pathway one — no pipeline, no material, no bind
/// group — so a filter producing one is still above the backends, in the way
/// [`DataArray`] is.
pub enum Product {
    Array(DataArray),
    Geometry(Mesh),
}

// Only a kind's own tests read a product back: [`collect`] takes them by value
// and matches, because it has to write each into a different place. These are
// how a test says "this output should have been an array" and gets told so by
// name rather than by a match arm that panics.
#[cfg(test)]
impl Product {
    /// The array, or `None` when this output is geometry.
    pub fn array(&self) -> Option<&DataArray> {
        match self {
            Product::Array(array) => Some(array),
            Product::Geometry(_) => None,
        }
    }

    /// The mesh, or `None` when this output is an array.
    pub fn geometry(&self) -> Option<&Mesh> {
        match self {
            Product::Geometry(mesh) => Some(mesh),
            Product::Array(_) => None,
        }
    }
}

impl From<DataArray> for Product {
    fn from(array: DataArray) -> Self {
        Product::Array(array)
    }
}

impl From<Mesh> for Product {
    fn from(mesh: Mesh) -> Self {
        Product::Geometry(mesh)
    }
}

/// What a run produced, keyed by [`OutputSpec::id`].
///
/// A run that cannot produce something — degenerate input, an unbound optional
/// it turned out to need — leaves it out rather than inventing an array. The
/// previous contents then stand, which is the honest outcome: nothing was
/// learned, so nothing changes.
pub type Products = HashMap<&'static str, Product>;

/// What each filter in a graph declares and what it has wired up, by handle:
/// its output specs, the handle it writes for each output, and the handle it
/// reads for each input.
///
/// One alias because two things build the same map — `source::WorldGraph`
/// from the live world, and `provenance`'s test double — and the tuple is wide
/// enough that spelling it twice invites the two to drift.
pub type Steps = HashMap<
    u64,
    (
        &'static [OutputSpec],
        HashMap<&'static str, u64>,
        HashMap<&'static str, u64>,
    ),
>;

/// What a run produced, and what went wrong if something did.
///
/// Products alone are not enough, because "produced nothing" is not one fact but
/// several: a `cartoon` with no backbone, a `gather` handed indices past the end
/// of its values, and a `contour` whose level sits outside the field are three
/// different mistakes that an empty output makes look identical. With arithmetic
/// in the graph a length mismatch between two arrays is the *routine* mistake,
/// and the user needs to be told which two lengths rather than left to guess why
/// a wire went dead.
///
/// A problem does **not** mean nothing was produced, and products do not mean
/// there was no problem: a filter may emit what it can and still say that an
/// input it wanted was unusable.
#[derive(Default)]
pub struct Outcome {
    pub products: Products,
    /// One sentence, addressed to whoever wired this up. `None` is success.
    pub problem: Option<String>,
}

impl Outcome {
    /// A run that produced nothing, for the stated reason.
    pub fn refused(why: impl Into<String>) -> Self {
        Self {
            products: Products::new(),
            problem: Some(why.into()),
        }
    }

    /// Products, plus a reason they are not the whole story.
    pub fn but(mut self, why: impl Into<String>) -> Self {
        self.problem = Some(why.into());
        self
    }

    /// Produced nothing, **and said why**.
    ///
    /// The conjunction is the point, and it is what a test should assert rather
    /// than emptiness alone. A run that produces nothing without a reason is the
    /// silent failure this type exists to abolish, so it must not satisfy the
    /// same assertion as a run that refused properly.
    #[cfg(test)]
    pub fn is_refusal(&self) -> bool {
        self.products.is_empty() && self.problem.is_some()
    }
}

/// So a filter that cannot fail — or has not yet been taught to say why — reads
/// as `products.into()` rather than naming a field.
impl From<Products> for Outcome {
    fn from(products: Products) -> Self {
        Self {
            products,
            problem: None,
        }
    }
}

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
    ///
    /// `None` marks a **source**: a kind whose output changes when an event
    /// happens rather than when its inputs do. [`start`] never schedules one —
    /// something else writes its [`Outputs`] directly, the same way [`collect`]
    /// would, which is what raises the `AssetEvent::Modified` that cascades
    /// downstream for free. Everything else about a source — creation, binding,
    /// rendering, staleness of what reads it — is unchanged, because a source
    /// still has `params`/`outputs` in the same shape a computed filter does.
    pub run: Option<fn(&Request) -> Outcome>,
}

impl FilterKind {
    /// Every input this kind binds data to, required or not.
    ///
    /// Only array inputs reach a run today — see [`Request`]. A filter reading
    /// another's *geometry* would have to copy a whole mesh onto the worker
    /// thread, and nothing wants that yet: the one filter that produces geometry
    /// takes arrays.
    pub fn inputs(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|spec| spec.kind.is_input())
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
        if let Some(existing) = self
            .kinds
            .iter_mut()
            .find(|existing| existing.id == kind.id)
        {
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
    task: Task<Outcome>,
    started_at: Generation,
}

/// Why this filter's last run did not do what was asked.
///
/// **Present means broken.** A marker carrying its reason, in the style of
/// [`Stale`] and [`Running`], rather than a status field that has to spell
/// "fine" — the healthy case is the absence of the component and costs nothing
/// to store or to check.
///
/// Removed on the first run that succeeds, so it never outlives the fault. It
/// survives a *discarded* run, though: a result thrown away for being stale
/// says nothing about whether the previous complaint still stands.
#[derive(Component, Debug, Clone)]
pub struct FilterProblem(pub String);

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
        let bus = FilterBus::from_world(app.world());
        app.insert_resource(bus).init_resource::<FilterRegistry>();
        {
            let mut registry = app.world_mut().resource_mut::<FilterRegistry>();
            cartoon::register(&mut registry);
            colormap::register(&mut registry);
            contour::register(&mut registry);
            geometry::register(&mut registry);
            index::register(&mut registry);
            maths::register(&mut registry);
            select::register(&mut registry);
            source::register(&mut registry);
        }

        app.configure_sets(
            Update,
            Filter.after(crate::scene::registry::apply_actor_params),
        )
        .add_systems(
            Update,
            (apply_filter_params, mark_stale, start, collect)
                .chain()
                .in_set(Filter),
        )
        // After the scene's own drain, so an array uploaded in this tick is
        // already in the store when a filter created in the same tick binds it.
        // Before `Filter`, so a filter created this tick runs this tick.
        .add_systems(
            Update,
            wire::apply_filter_commands
                .after(crate::scene::apply_scene_commands)
                .before(Filter),
        );

        // `on_click` is a Bevy observer, not a scheduled system, so it is not
        // ordered against here — pointer observers run before `Update`
        // regardless, and `Picked` persists across two frames, so this only
        // has to land before the frame is over for `AssetEvent::Modified` to
        // be seen next frame the way any other filter's write is.
        // `add_message` is idempotent — `ScenePlugin` owns the type and also
        // registers it — and is repeated because this module's test harness
        // builds a minimal `App` with `FilterPlugin` alone, and
        // `write_picks`'s `MessageReader<Picked>` panics on an unregistered
        // message type rather than silently reading nothing.
        app.add_message::<crate::scene::Picked>()
            .add_systems(Update, source::write_picks.before(Filter));
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
                    .array(*id)
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
        // A source has nothing for a worker thread to do: whatever wrote its
        // `Outputs` already raised the `AssetEvent::Modified` this filter would
        // have raised itself. `Stale` is cleared here rather than left set,
        // which would otherwise show as a permanent "pending" spinner
        // (`ui/gather.rs` reads `Has<Stale>`) between one event and the next.
        let Some(run) = registered.run else {
            commands.entity(entity).remove::<Stale>();
            continue;
        };

        let mut inputs = HashMap::new();
        for spec in registered.inputs() {
            let Some(handle) = bound.get(spec.id) else {
                continue;
            };
            let Some(array) = store
                .array(handle)
                .and_then(|held| arrays.get(&held.handle))
            else {
                continue;
            };
            inputs.insert(spec.id, array.clone());
        }

        let request = Request {
            params: params.0.clone(),
            inputs,
        };
        commands.entity(entity).remove::<Stale>().insert(Running {
            task: pool.spawn(async move { run(&request) }),
            started_at: *generation,
        });
    }
}

/// Writes a finished run into the output arrays and meshes.
///
/// Rewrites the existing assets rather than replacing the handles, which is what
/// keeps a consumer's binding valid and what raises the `AssetEvent::Modified`
/// everything downstream is watching.
///
/// A geometry output rewrites a `Mesh` the same way, and that is the whole reach
/// of it: an actor already holds a `Mesh3d` naming that asset, so Bevy re-uploads
/// the new vertices under it with nothing in this module involved. Nothing
/// rebuilds and nothing rebinds.
fn collect(
    mut commands: Commands,
    mut arrays: ResMut<Assets<DataArray>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut store: ResMut<DataStore>,
    mut running: Query<(Entity, &mut Running, &Generation, &Outputs, &FilterKindId)>,
) {
    for (entity, mut run, generation, outputs, kind) in &mut running {
        let Some(outcome) = block_on(future::poll_once(&mut run.task)) else {
            continue;
        };
        commands.entity(entity).remove::<Running>();

        // Stale before it finished. Marking it again would be wrong — it is
        // already marked, which is why the generation moved — so this only
        // drops the answer. The previous problem, if any, is left standing:
        // a discarded run learned nothing either way.
        if run.started_at != *generation {
            continue;
        }

        // Said before the products are written, so a filter that produced
        // something *and* complained keeps both.
        match &outcome.problem {
            Some(why) => {
                warn!("filter: {} {why}", kind.0);
                commands.entity(entity).insert(FilterProblem(why.clone()));
            }
            None => {
                commands.entity(entity).remove::<FilterProblem>();
            }
        }

        for (output, produced) in outcome.products {
            let Some(handle) = outputs.get(output) else {
                warn!("filter: {} produced undeclared output \"{output}\"", kind.0);
                continue;
            };
            match produced {
                Product::Array(produced) => {
                    let Some(held) = store.array(handle) else {
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
                    // The meta as well as the bytes: an output's length is
                    // decided by the run, so a consumer asking what shape this
                    // handle is has to be told the new answer rather than the
                    // one it was created with.
                    store.insert(handle, meta, asset);
                }
                Product::Geometry(produced) => {
                    let Some(held) = store.geometry(handle) else {
                        continue;
                    };
                    let meta = describe(output, &produced);
                    let asset = held.handle.clone();
                    let Some(mut existing) = meshes.get_mut(&asset) else {
                        continue;
                    };
                    *existing = produced;
                    store.insert_geometry(handle, meta, asset);
                }
            }
        }
    }
}

/// What a finished mesh carries, as a consumer will be told it.
///
/// Read off the mesh rather than declared by the kind: which attributes a run
/// produced depends on what was bound to it, so `cartoon` with no colours bound
/// and `cartoon` with them are the same kind producing different geometry.
pub(crate) fn describe(name: &str, mesh: &Mesh) -> crate::scene::data::GeometryMeta {
    crate::scene::data::GeometryMeta {
        name: name.to_string(),
        vertices: mesh.count_vertices() as u64,
        // Triangles rather than indices, because that is what a caller counts.
        // An unindexed mesh has none and its vertices are its corners.
        triangles: mesh
            .indices()
            .map_or(mesh.count_vertices() / 3, |indices| indices.len() / 3)
            as u64,
        normals: mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some(),
        colours: mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some(),
    }
}

#[cfg(test)]
mod tests;
