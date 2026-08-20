//! One pass over the world, flattened into rows the tabs can borrow freely.
//!
//! egui builds widgets inside closures, and several of them want the same world
//! data at once. Reading everything up front into plain owned rows is what lets
//! a tab take `&Row` wherever it likes; threading Bevy queries down through
//! nested closures instead means a borrow conflict at every level.
//!
//! An object's children are two things at once — nested objects and the
//! placements drawn under it — and the rows split them apart, by what each
//! entity carries rather than by any second link. A placement resolves to the
//! actor it is a copy of, so one actor drawn under three objects appears in
//! three rows and is the same row content each time.

use bevy::asset::AssetId;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::counter::UniqueID;
use crate::filter::{
    FilterKindId, FilterParams, FilterProblem, FilterRegistry, OutputSpec, Outputs, Running, Stale,
};
use crate::scene::registry::{
    ActorKindId, ActorParams, ActorRegistry, ParamMap, ParamSpec, data as bound_handle,
};
use crate::scene::{BufferMeta, DataStore, HeldMeta, Parents, Placement};
use crate::scene::{DataArray, SceneObject};

/// A flattened view of one object.
pub struct Row {
    pub entity: Entity,
    pub id: u64,
    pub name: String,
    pub visible: bool,
    /// Everything drawn under this object.
    pub actors: Vec<ActorRow>,
    /// Kinds that could be added to this object. Resolved while gathering so
    /// the drawing closures never borrow the registry.
    pub available: Vec<KindOption>,
    /// Child *objects* only. Actors are children too and are excluded here.
    pub children: Vec<Entity>,
}

/// One kind offered in the "draw this another way" picker.
pub struct KindOption {
    pub id: &'static str,
    pub label: &'static str,
}

pub struct ActorRow {
    pub entity: Entity,
    pub id: u64,
    pub label: &'static str,
    /// The controls to show, taken straight from the backend's declaration —
    /// `&'static` so nothing here has to be cloned or borrowed from the world.
    pub specs: &'static [ParamSpec],
    pub params: ParamMap,
    /// How many objects draw it. More than one means editing it here changes
    /// every copy, which is worth saying where the controls are.
    pub places: usize,
}

/// One filter as a row.
///
/// The mirror of [`ActorRow`], and deliberately so: a filter kind declares
/// `ParamSpec`s exactly as an actor kind does, which is what lets one set of
/// generated controls serve both. What a filter has instead of a place in the
/// tree is [`outputs`](Self::outputs) — it is reached through the data it
/// writes, not through anything it is drawn under.
pub struct FilterRow {
    pub entity: Entity,
    pub id: u64,
    pub kind: &'static str,
    pub label: &'static str,
    pub specs: &'static [ParamSpec],
    pub params: ParamMap,
    /// What it writes, in declaration order, as `(output id, handle)`.
    ///
    /// Allocated when the filter was created rather than when it first ran, so
    /// these are bindable before a single value exists behind them.
    pub outputs: Vec<(&'static OutputSpec, u64)>,
    /// A run is in flight.
    pub busy: bool,
    /// The results no longer follow from the inputs.
    ///
    /// Worth showing beside `busy` rather than folding the two together: a chain
    /// costs a frame per link, so a filter three deep sits stale-but-not-yet-busy
    /// for a moment after every edit, and that is normal rather than stuck.
    pub stale: bool,
    /// Why the last run did not do what was asked, if it did not.
    ///
    /// Distinct from the refusal a *creation* can carry: that one is about a
    /// call, is reported once and goes at the foot of the panel. This is a
    /// standing property of the filter itself, so it travels on the row and
    /// shows wherever the filter does.
    pub problem: Option<String>,
}

/// Who holds an array, and what reads it.
pub struct Owner {
    pub object: u64,
    /// The buffer name it was uploaded under.
    pub name: String,
}

/// Something that reads a handle: the filter or actor, and which input.
pub struct Consumer {
    /// The reader's own handle, for naming it.
    pub id: u64,
    pub label: &'static str,
    pub input: &'static str,
}

/// Everything the UI reads about objects, in one query.
///
/// A type alias rather than nine parameters repeated at each call site; Bevy
/// accepts it as a system parameter unchanged.
pub type ObjectData<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static SceneObject,
        &'static Visibility,
        Option<&'static Children>,
        Option<&'static ChildOf>,
    ),
>;

/// Everything the UI reads about actors. See [`ObjectData`].
pub type ActorData<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static ActorKindId,
        &'static ActorParams,
        &'static Parents,
    ),
>;

/// Everything the UI reads about filters. See [`ObjectData`].
///
/// Read-only on purpose, rather than [`filter::Filters`](crate::filter::Filters)
/// — that wraps a mutable query over the same components and would conflict with
/// every other reader in the schedule. The UI never writes a filter directly
/// anyway; it asks through [`SceneCommand`](crate::scene::SceneCommand).
///
/// [`Running`] has private fields, so `Has` is as much as can be learned about a
/// run in flight from out here. It is enough: the question the panel asks is
/// "is this one busy", not "since when".
pub type FilterData<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static UniqueID,
        &'static FilterKindId,
        &'static FilterParams,
        &'static Outputs,
        Has<Stale>,
        Has<Running>,
        Option<&'static FilterProblem>,
    ),
    With<FilterKindId>,
>;

/// The whole scene as the UI sees it for one frame.
///
/// `Default` is an empty scene — nothing uploaded and nothing drawn, which is a
/// real state and the one the control tests draw against.
#[derive(Default)]
pub struct Gathered {
    pub rows: HashMap<Entity, Row>,
    /// Objects with no object parent, in handle order.
    pub roots: Vec<Entity>,
    /// Every object in handle order, which is how the Data and Actors tabs
    /// group their listings.
    pub ordered: Vec<Entity>,
    /// Actors drawn under no object, in handle order.
    ///
    /// Deleting an object costs an actor that placement rather than its life,
    /// so losing the last one leaves it defined but nowhere. It appears in no
    /// object's row then — without a list of its own there would be no way to
    /// reach it, or to tell it still exists.
    pub detached: Vec<ActorRow>,
    pub owners: HashMap<AssetId<DataArray>, Owner>,
    /// Arrays uploaded on their own, as the handle a client knows them by and
    /// the label it sent. No object holds these, so `owners` says nothing about
    /// them — without this they would be listed as unreferenced, which is the
    /// opposite of true.
    pub held: HashMap<AssetId<DataArray>, (u64, BufferMeta)>,
    /// Everything a client holds, in handle order, for the input pickers.
    /// Arrays and meshes together, because an input decides for itself which it
    /// takes — see [`ParamKind::accepts`](crate::scene::registry::ParamKind).
    pub bindable: Vec<(u64, HeldMeta)>,
    /// Every filter, in handle order. Flat, because a filter belongs to no
    /// object — it is reached through the data it writes.
    pub filters: Vec<FilterRow>,
    /// Data handle → the filter that writes it.
    ///
    /// What turns a bare `d12` on an input row into `d12 · from [3] colour map`,
    /// which is the whole of a chain's legibility until the node editor exists.
    /// Built straight off each filter's `Outputs`; no `Graph` is needed for it.
    pub producers: HashMap<u64, u64>,
    /// Data handle → everything that reads it, filters and actors alike.
    ///
    /// The other direction of the same question, and the one that answers "is it
    /// safe to remove this filter" — an output nothing reads is a dead branch,
    /// and the panel should say so rather than leave it looking connected.
    pub consumers: HashMap<u64, Vec<Consumer>>,
    pub total_bytes: u64,
    /// Meshes a filter assembled, and the vertices across them.
    ///
    /// Counted separately from the arrays because a vertex is on the GPU rather
    /// than in `Assets<DataArray>`, and because this is the number that answers
    /// "did sharing the geometry work": drawing one ribbon two ways should move
    /// neither of these.
    pub meshes: usize,
    pub vertices: u64,
}

impl Gathered {
    /// The actor with this entity, together with the object it is drawn under.
    ///
    /// The object comes back too, because the controls head their panel with
    /// which object the actor is drawn under — and it is optional because a
    /// detached actor has none to name.
    pub fn actor(&self, entity: Entity) -> Option<(Option<&Row>, &ActorRow)> {
        let attached = self.rows.values().find_map(|row| {
            row.actors
                .iter()
                .find(|actor| actor.entity == entity)
                .map(|actor| (Some(row), actor))
        });
        attached.or_else(|| {
            self.detached
                .iter()
                .find(|actor| actor.entity == entity)
                .map(|actor| (None, actor))
        })
    }

    /// By handle rather than entity, as everything else that names a filter
    /// does. It also sidesteps a race: a filter is created through a command, and
    /// its reply arrives before the spawn has been applied, so there is a moment
    /// when the handle exists and the entity does not.
    pub fn filter(&self, id: u64) -> Option<&FilterRow> {
        self.filters.iter().find(|row| row.id == id)
    }

    /// The handles of every object an actor is drawn under.
    ///
    /// Rebuilt from the rows rather than read off `Parents`, because the node
    /// canvas needs the *set* in order to send it back one member longer or
    /// shorter — `SetActor` replaces the whole thing, so adding one edge means
    /// knowing all of them.
    pub fn objects_of(&self, actor: Entity) -> Vec<u64> {
        let mut objects: Vec<u64> = self
            .rows
            .values()
            .filter(|row| row.actors.iter().any(|drawn| drawn.entity == actor))
            .map(|row| row.id)
            .collect();
        objects.sort_unstable();
        objects
    }

    /// How to name a bound handle on an input row.
    ///
    /// `d12 colour · from [3] colour map` when a filter writes it, `d4 positions`
    /// when it was uploaded. Saying where data came from is what makes a chain
    /// readable in a flat list; without it every input is an opaque number.
    pub fn describe_handle(&self, handle: u64) -> String {
        let name = self
            .bindable
            .iter()
            .find(|(id, _)| *id == handle)
            .map(|(_, meta)| meta.name())
            .unwrap_or("<gone>");
        match self
            .producers
            .get(&handle)
            .and_then(|producer| self.filters.iter().find(|row| row.id == *producer))
        {
            Some(from) => format!("d{handle} {name} · from [{}] {}", from.id, from.label),
            None => format!("d{handle} {name}"),
        }
    }
}

/// One actor as a row, or `None` if the entity is not an actor.
///
/// A kind with no registration cannot be drawn or configured, so there is
/// nothing useful to show for it either.
fn actor_row(actors: &ActorData, registry: &ActorRegistry, entity: Entity) -> Option<ActorRow> {
    let (entity, id, kind, params, parents) = actors.get(entity).ok()?;
    let registered = registry.get(kind.0)?;
    Some(ActorRow {
        entity,
        id: id.0,
        label: registered.label,
        specs: registered.params,
        params: params.0.clone(),
        places: parents.0.len(),
    })
}

/// Everything one pass over the world reads, as a single system parameter.
///
/// Seven of them, and they always travel together — this *is* [`gather`]'s
/// argument list. Bundling them is not only tidiness: `draw_ui` was at
/// seventeen parameters against Bevy's limit of sixteen, and a system that goes
/// one over fails with "does not describe a valid system configuration", which
/// names none of the parameters involved.
#[derive(bevy::ecs::system::SystemParam)]
pub struct SceneRead<'w, 's> {
    pub objects: ObjectData<'w, 's>,
    pub actors: ActorData<'w, 's>,
    pub filters: FilterData<'w, 's>,
    pub placements: Query<'w, 's, &'static Placement>,
    pub registry: Res<'w, ActorRegistry>,
    pub filter_registry: Res<'w, FilterRegistry>,
    pub store: Res<'w, DataStore>,
}

pub fn gather(read: &SceneRead) -> Gathered {
    let SceneRead {
        objects,
        actors,
        filters,
        placements,
        registry,
        filter_registry,
        store,
    } = read;
    let mut rows: HashMap<Entity, Row> = HashMap::new();
    let mut roots: Vec<Entity> = Vec::new();
    let owners: HashMap<AssetId<DataArray>, Owner> = HashMap::new();
    let held: HashMap<AssetId<DataArray>, (u64, BufferMeta)> = store
        .iter()
        .map(|(id, array)| (array.handle.id(), (id, array.meta.clone())))
        .collect();
    // The same, plus the meshes, as a list an input picker can walk in handle
    // order.
    let mut bindable: Vec<(u64, HeldMeta)> = store
        .iter()
        .map(|(id, array)| (id, HeldMeta::Array(array.meta.clone())))
        .chain(
            store
                .iter_geometry()
                .map(|(id, mesh)| (id, HeldMeta::Geometry(mesh.meta.clone()))),
        )
        .collect();
    bindable.sort_by_key(|(id, _)| *id);

    for (entity, id, object, visibility, children, parent) in objects {
        // One child list, told apart by what each entity carries: a child that
        // is an object is a nested node, and one the actor query matches is
        // something drawn here.
        let child_objects: Vec<Entity> = children
            .into_iter()
            .flatten()
            .copied()
            .filter(|child| objects.contains(*child))
            .collect();

        // Each placement stands for an actor. Resolved to the actor here, so a
        // row shows the drawing itself — one set of controls, whichever of its
        // objects it is being looked at under.
        let drawn: Vec<ActorRow> = children
            .into_iter()
            .flatten()
            .copied()
            .filter_map(|child| actor_row(actors, registry, placements.get(child).ok()?.0))
            .collect();

        // Every kind, for every object. What an actor draws is what it binds, so
        // there is nothing about this node that could rule a kind out.
        let available: Vec<KindOption> = registry
            .iter()
            .map(|kind| KindOption {
                id: kind.id,
                label: kind.label,
            })
            .collect();

        // An object owns no arrays at all: every array is a handle the store
        // knows about, whoever made it.

        // A parent that is not itself an object does not make this a child.
        let parented = parent.is_some_and(|link| objects.contains(link.parent()));
        if !parented {
            roots.push(entity);
        }

        rows.insert(
            entity,
            Row {
                entity,
                id: id.0,
                name: object.name.clone(),
                visible: *visibility != Visibility::Hidden,
                actors: drawn,
                available,
                children: child_objects,
            },
        );
    }

    // Actors with nowhere to be drawn. Asked of the parent list rather than of
    // the tree, because an actor is not in the tree — only its placements are.
    let mut detached: Vec<ActorRow> = actors
        .iter()
        .filter(|(.., parents)| parents.0.is_empty())
        .filter_map(|(entity, ..)| actor_row(actors, registry, entity))
        .collect();
    detached.sort_by_key(|row| row.id);

    let handle = |entity: &Entity| rows.get(entity).map(|row| row.id).unwrap_or(u64::MAX);
    roots.sort_by_key(handle);
    let mut ordered: Vec<Entity> = rows.keys().copied().collect();
    ordered.sort_by_key(handle);

    // Filters, and the two directions of the graph they make between them.
    let mut filter_rows: Vec<FilterRow> = Vec::new();
    let mut producers: HashMap<u64, u64> = HashMap::new();
    let mut consumers: HashMap<u64, Vec<Consumer>> = HashMap::new();

    for (entity, id, kind, params, outputs, stale, busy, problem) in filters {
        let Some(registered) = filter_registry.get(kind.0) else {
            // A kind with no registration cannot run or be configured, so there
            // is nothing useful to show for it.
            continue;
        };
        for spec in registered.outputs {
            if let Some(handle) = outputs.get(spec.id) {
                producers.insert(handle, id.0);
            }
        }
        filter_rows.push(FilterRow {
            entity,
            id: id.0,
            kind: registered.id,
            label: registered.label,
            specs: registered.params,
            params: params.0.clone(),
            // Walked over the kind's *declaration* rather than over the
            // component's map, so the order is the one the kind states and holds
            // still frame to frame. Iterating the `HashMap` would reshuffle the
            // rows under the pointer.
            outputs: registered
                .outputs
                .iter()
                .filter_map(|spec| Some((spec, outputs.get(spec.id)?)))
                .collect(),
            busy,
            stale,
            problem: problem.map(|problem| problem.0.clone()),
        });
    }
    filter_rows.sort_by_key(|row| row.id);

    // Who reads what. Filters and actors together — an output feeding a `surface`
    // is exactly as much a consumer as one feeding another filter, and the panel
    // asks the question the same way for both.
    let mut record = |params: &ParamMap, specs: &'static [ParamSpec], reader: Consumer| {
        for spec in specs.iter().filter(|spec| spec.kind.is_input()) {
            if let Some(handle) = bound_handle(params, spec.id) {
                consumers.entry(handle).or_default().push(Consumer {
                    input: spec.id,
                    ..reader
                });
            }
        }
    };
    for row in &filter_rows {
        record(
            &row.params,
            row.specs,
            Consumer {
                id: row.id,
                label: row.label,
                input: "",
            },
        );
    }
    for (_, id, kind, params, ..) in actors {
        let Some(registered) = registry.get(kind.0) else {
            continue;
        };
        record(
            &params.0,
            registered.params,
            Consumer {
                id: id.0,
                label: registered.label,
                input: "",
            },
        );
    }

    // Every array in memory. They are all held rather than owned now, so there
    // is no second source to add in.
    let total_bytes: u64 = store
        .iter()
        .filter_map(|(_, array)| array.meta.byte_length())
        .sum();

    Gathered {
        rows,
        roots,
        ordered,
        detached,
        owners,
        held,
        bindable,
        filters: filter_rows,
        producers,
        consumers,
        total_bytes,
        meshes: store.iter_geometry().count(),
        vertices: store
            .iter_geometry()
            .map(|(_, mesh)| mesh.meta.vertices)
            .sum(),
    }
}
