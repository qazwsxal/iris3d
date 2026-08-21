//! The actor commands: adding, configuring, removing and listing.
//!
//! An actor is one way of drawing something. It binds arrays to the inputs its
//! kind declares, and is placed under the objects it is drawn under — two
//! separate questions, which is why binding is checked here and placement is
//! left to `link`.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::counter::{GlobalIDCounter, UniqueID};
use crate::data::DataStore;
use crate::model::SceneError;

use super::apply::{ActorItem, ActorQuery};
use super::link::{self, Parents, Shown};
use super::object_commands::spawn_object;
use super::registry::{self, ActorKindId, ActorParams, ActorRegistry};
use super::{ActorSummary, KindSummary, SceneObject};

/// The binding gate for an actor kind. See [`crate::model::check_bindings`],
/// which an actor and a filter share.
pub(crate) fn check_bindings(
    kind: &registry::ActorKind,
    params: &crate::model::ParamMap,
    store: &DataStore,
) -> Result<(), SceneError> {
    crate::model::check_bindings(kind.id, kind.inputs(), params, store)
}

/// Adds a way of drawing, at any number of places in the tree.
///
/// `parents` may be empty, in which case an object is made to hold this actor.
/// An actor has no place of its own, so it needs at least one — and a caller
/// with nothing to group it under would otherwise open every single drawing
/// with the same `CreateObject` call.
///
/// Naming several draws one actor in several places. It stays one actor: one
/// mesh, one set of parameters, and every copy changes together.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_actor(
    commands: &mut Commands,
    counter: &mut GlobalIDCounter,
    registry: &ActorRegistry,
    index: &mut HashMap<u64, Entity>,
    drawn: &mut HashMap<u64, Entity>,
    parents: Vec<u64>,
    kind: String,
    params: crate::model::ParamMap,
    store: &DataStore,
) -> Result<ActorSummary, SceneError> {
    // Resolved before anything is created, so a bad handle builds nothing.
    let named: Vec<(u64, Entity)> = parents
        .into_iter()
        .map(|id| {
            index
                .get(&id)
                .copied()
                .map(|entity| (id, entity))
                .ok_or(SceneError::NoSuchObject(id))
        })
        .collect::<Result<_, _>>()?;

    // The caller names the kind. There is no default to fall back on: which
    // representation suits some data is a judgement, and the server has no
    // basis for it beyond the order the kinds happened to register in.
    //
    // Nor is there a check that the kind suits the object. An actor binds its
    // own arrays, so what the object holds — usually nothing — says nothing
    // about what can draw there. `check_bindings` is the real gate, and it asks
    // about the data rather than about the node.
    let registered = registry.get(&kind).ok_or_else(|| SceneError::UnknownKind {
        kind: kind.clone(),
        backend: registry.backend(),
    })?;

    // Unset parameters take the kind's default: this is a new actor, so there
    // is no previous value to preserve.
    let params = registered.normalise(&params);
    check_bindings(registered, &params, store)?;

    // Only now, with every reason to refuse behind us. Creating the object any
    // earlier would leave an empty one behind whenever an actor is rejected.
    let placed = match named.is_empty() {
        false => named,
        true => {
            let name = registered.label.to_string();
            let (id, _) = spawn_object(commands, counter, index, SceneObject { name });
            vec![(id, index[&id])]
        }
    };
    let (parents, entities): (Vec<u64>, Vec<Entity>) = placed.into_iter().unzip();

    let (id, entity) = link::spawn_actor(
        commands,
        counter,
        entities,
        (ActorKindId(registered.id), ActorParams(params.clone())),
    );
    drawn.insert(id, entity);

    info!(
        "scene: actor {id} draws {} under object(s) {parents:?}",
        registered.id
    );

    Ok(ActorSummary {
        id,
        kind: registered.id.to_string(),
        parents,
        params,
        visible: true,
    })
}

/// Changes an existing actor, leaving anything unnamed alone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn set_actor(
    registry: &ActorRegistry,
    drawn: &HashMap<u64, Entity>,
    index: &HashMap<u64, Entity>,
    actors: &mut ActorQuery,
    ids: &Query<&UniqueID>,
    store: &DataStore,
    id: u64,
    params: crate::model::ParamMap,
    visible: Option<bool>,
    parents: Option<Vec<u64>>,
) -> Result<ActorSummary, SceneError> {
    let entity = *drawn.get(&id).ok_or(SceneError::NoSuchActor(id))?;

    // Resolved before anything is written, so a bad handle changes nothing.
    let moving_to = parents
        .map(|wanted| {
            wanted
                .into_iter()
                .map(|id| index.get(&id).copied().ok_or(SceneError::NoSuchObject(id)))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    let Ok(mut item) = actors.get_mut(entity) else {
        // In `drawn` but not yet in the world: added earlier in this same
        // drain, so the query has not seen it. Nothing is lost by asking again
        // next tick, and guessing would mean writing to an entity blind.
        return Err(SceneError::NoSuchActor(id));
    };

    let registered = registry
        .get(item.2.0)
        .ok_or_else(|| SceneError::UnknownKind {
            kind: item.2.0.to_string(),
            backend: registry.backend(),
        })?;

    // Merge rather than replace: a client changing one setting should not have
    // to restate the others, and omitting them must not silently reset them.
    // Merged into a copy, because the bindings are judged as a whole below and
    // a refusal has to leave the actor exactly as it was.
    let mut merged = item.3.0.clone();
    for (key, value) in params {
        let Some(value) = registered
            .spec(&key)
            .and_then(|spec| spec.kind.sanitise(value))
        else {
            warn!("scene: actor {id} has no parameter \"{key}\" of that type");
            continue;
        };
        // Clearing is a removal, not a value — see the same merge in
        // `filter::wire::set`. An input let go has to be absent from the map,
        // because absence is what every reader already understands as unbound.
        match value {
            crate::model::ParamValue::Unset => merged.remove(&key),
            value => merged.insert(key, value),
        };
    }

    // The same gate `add_actor` passes through. `sanitise` only says that a
    // handle is a handle; whether that particular array fits the input needs
    // the store, so without this a rebind to the wrong dtype or shape would be
    // refused when adding an actor and accepted when changing one.
    check_bindings(registered, &merged, store)?;
    // Written only when it differs, so a call that restates the parameters does
    // not mark them `Changed` and throw the geometry away to rebuild it
    // identically.
    item.3.set_if_neq(ActorParams(merged));

    if let Some(visible) = visible {
        *item.4 = Shown(visible);
    }

    if let Some(wanted) = moving_to {
        // No cycle check, unlike `set_parent`. Nothing is ever parented under
        // a placement, so no arrangement of them can close a loop.
        //
        // A plain write, not a queued command: `sync_placements` reads this
        // next and builds or drops placements to match, so adding a parent and
        // taking the last one away are the same operation.
        *item.5 = Parents(wanted);
    }

    summarise_actor(
        actors.get(entity).expect("the entity was just written"),
        ids,
    )
    .ok_or(SceneError::NoSuchActor(id))
}

/// Describes one actor from its query item.
///
/// `Option` only so callers can filter with `?`; an actor always describes.
pub(crate) fn summarise_actor(
    (_, id, kind, params, shown, parents): ActorItem<'_>,
    ids: &Query<&UniqueID>,
) -> Option<ActorSummary> {
    // Read straight off the component, unlike an object's parent. Placements
    // are built from this by a later system rather than by a queued command, so
    // there is no deferred write for a listing in this same drain to miss.
    let mut parents: Vec<u64> = parents
        .0
        .iter()
        .filter_map(|parent| ids.get(*parent).ok())
        .map(|unique| unique.0)
        .collect();
    parents.sort_unstable();

    Some(ActorSummary {
        id: id.0,
        kind: kind.0.to_string(),
        parents,
        params: params.0.clone(),
        // What this actor was told. An object hidden above one of its
        // placements still hides that copy, which is `InheritedVisibility`'s
        // business and not reported here.
        visible: shown.0,
    })
}

/// Destroys an actor. Reports whether there was one.
///
/// Actors own nothing, so there is no subtree question of the kind object
/// deletion has to answer.
pub(crate) fn remove_actor(
    commands: &mut Commands,
    drawn: &mut HashMap<u64, Entity>,
    id: u64,
) -> bool {
    match drawn.remove(&id) {
        Some(entity) => {
            commands.entity(entity).despawn();
            info!("scene: removed actor {id}");
            true
        }
        None => false,
    }
}

/// Every actor, optionally restricted to those drawn under one object.
///
/// Filtered on where an actor is drawn, not on what it reads: an actor reads
/// arrays, and any number of them.
pub(crate) fn list_actors(
    actors: &ActorQuery,
    ids: &Query<&UniqueID>,
    index: &HashMap<u64, Entity>,
    parent: Option<u64>,
) -> Result<Vec<ActorSummary>, SceneError> {
    let filter = match parent {
        Some(id) => Some(*index.get(&id).ok_or(SceneError::NoSuchObject(id))?),
        None => None,
    };
    let mut listing: Vec<ActorSummary> = actors
        .iter()
        .filter(|item| filter.is_none_or(|object| item.5.0.contains(&object)))
        .filter_map(|item| summarise_actor(item, ids))
        .collect();
    listing.sort_by_key(|summary| summary.id);
    Ok(listing)
}

/// What the running backend can draw, in registration order.
pub(crate) fn list_actor_kinds(registry: &ActorRegistry) -> Vec<KindSummary> {
    registry
        .iter()
        .map(|kind| KindSummary {
            id: kind.id.to_string(),
            label: kind.label.to_string(),
            params: kind.params,
        })
        .collect()
}
