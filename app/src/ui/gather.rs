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
use crate::scene::registry::{ActorKindId, ActorParams, ActorRegistry, ParamMap, ParamSpec};
use crate::scene::{BufferMeta, DataStore, HeldMeta, Parents, Placement};
use crate::scene::{DataArray, SceneObject, Subset};

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
    pub subset: Subset,
    /// How many objects draw it. More than one means editing it here changes
    /// every copy, which is worth saying where the controls are.
    pub places: usize,
}

/// Who holds an array, and what reads it.
pub struct Owner {
    pub object: u64,
    /// The buffer name it was uploaded under.
    pub name: String,
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
        &'static Subset,
        &'static Parents,
    ),
>;

/// The whole scene as the UI sees it for one frame.
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
}

/// One actor as a row, or `None` if the entity is not an actor.
///
/// A kind with no registration cannot be drawn or configured, so there is
/// nothing useful to show for it either.
fn actor_row(actors: &ActorData, registry: &ActorRegistry, entity: Entity) -> Option<ActorRow> {
    let (entity, id, kind, params, subset, parents) = actors.get(entity).ok()?;
    let registered = registry.get(kind.0)?;
    Some(ActorRow {
        entity,
        id: id.0,
        label: registered.label,
        specs: registered.params,
        params: params.0.clone(),
        subset: subset.clone(),
        places: parents.0.len(),
    })
}

pub fn gather(
    objects: &ObjectData,
    actors: &ActorData,
    placements: &Query<&Placement>,
    registry: &ActorRegistry,
    store: &DataStore,
) -> Gathered {
    let mut rows: HashMap<Entity, Row> = HashMap::new();
    let mut roots: Vec<Entity> = Vec::new();
    let mut owners: HashMap<AssetId<DataArray>, Owner> = HashMap::new();
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

        // An object owns no arrays now. The only thing still tied to one is an
        // actor's selection, which the actor holds rather than the store —
        // without this the inventory calls it unreferenced, which is backwards.
        for actor in &drawn {
            if let Subset::Selected { array, .. } = &actor.subset {
                owners.insert(
                    array.id(),
                    Owner {
                        object: id.0,
                        name: "subset".into(),
                    },
                );
            }
        }

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
        total_bytes,
        meshes: store.iter_geometry().count(),
        vertices: store.iter_geometry().map(|(_, mesh)| mesh.meta.vertices).sum(),
    }
}
