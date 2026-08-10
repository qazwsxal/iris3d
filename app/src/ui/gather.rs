//! One pass over the world, flattened into rows the tabs can borrow freely.
//!
//! egui builds widgets inside closures, and several of them want the same world
//! data at once. Reading everything up front into plain owned rows is what lets
//! a tab take `&Row` wherever it likes; threading Bevy queries down through
//! nested closures instead means a borrow conflict at every level.
//!
//! The rows are also where the two different questions about an object get
//! answered separately: `children` is the transform tree, `actors` is what
//! draws this object's data. Those are no longer the same list split in half —
//! an actor may sit under some entirely different object.

use bevy::asset::AssetId;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::counter::UniqueID;
use crate::scene::data::{FieldKind, Fields};
use crate::scene::{BufferMeta, DataStore};
use crate::scene::registry::{ActorKindId, ActorParams, ActorRegistry, ParamMap, ParamSpec};
use crate::scene::{Actors, ColorBy, DataArray, DatasetKind, SceneObject, Subset};

/// A flattened view of one object.
pub struct Row {
    pub entity: Entity,
    pub id: u64,
    pub name: String,
    pub kind: DatasetKind,
    pub visible: bool,
    pub arrays: usize,
    pub bytes: u64,
    /// Field name and what shape its values have, for the colour-by picker.
    pub fields: Vec<FieldRow>,
    /// Everything drawing this object's data, wherever it is placed.
    pub actors: Vec<ActorRow>,
    /// Kinds that could be added to this object, as `(id, label)`. Resolved
    /// while gathering so the drawing closures never borrow the registry.
    pub available: Vec<(&'static str, &'static str)>,
    /// Child *objects* only. Actors are children too and are excluded here.
    pub children: Vec<Entity>,
}

pub struct FieldRow {
    pub name: String,
    pub kind: &'static str,
}

pub struct ActorRow {
    pub entity: Entity,
    pub id: u64,
    pub label: &'static str,
    /// The controls to show, taken straight from the backend's declaration —
    /// `&'static` so nothing here has to be cloned or borrowed from the world.
    pub specs: &'static [ParamSpec],
    pub params: ParamMap,
    pub colour: ColorBy,
    pub subset: Subset,
}

/// Who holds an array, and what reads it.
pub struct Owner {
    pub object: u64,
    /// The buffer name it was uploaded under.
    pub name: String,
    /// Fields whose values are these bytes. Separate from `name` because a
    /// buffer is a field as well as a buffer, and the two names can differ.
    pub fields: Vec<String>,
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
        &'static DatasetKind,
        &'static Visibility,
        Option<&'static Fields>,
        Option<&'static Children>,
        Option<&'static Actors>,
        Option<&'static ChildOf>,
    ),
>;

/// Everything the UI reads about actors. See [`ObjectData`].
pub type ActorData<'w, 's> = Query<
    'w,
    's,
    (
        &'static UniqueID,
        &'static ActorKindId,
        &'static ActorParams,
        &'static ColorBy,
        &'static Subset,
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
    pub owners: HashMap<AssetId<DataArray>, Owner>,
    /// Arrays uploaded on their own, as the handle a client knows them by and
    /// the label it sent. No object holds these, so `owners` says nothing about
    /// them — without this they would be listed as unreferenced, which is the
    /// opposite of true.
    pub held: HashMap<AssetId<DataArray>, (u64, BufferMeta)>,
    /// Every held array in handle order, for the input pickers.
    pub bindable: Vec<(u64, BufferMeta)>,
    pub total_bytes: u64,
}

impl Gathered {
    /// The actor with this entity, together with the object whose data it
    /// draws.
    ///
    /// The pair comes back together because every control that edits an actor
    /// also needs the source's fields — the colour picker offers them, and so
    /// does any `Field` parameter.
    pub fn actor(&self, entity: Entity) -> Option<(&Row, &ActorRow)> {
        self.rows.values().find_map(|row| {
            row.actors
                .iter()
                .find(|actor| actor.entity == entity)
                .map(|actor| (row, actor))
        })
    }
}

pub fn gather(
    objects: &ObjectData,
    actors: &ActorData,
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
    // The same arrays as a list an input picker can walk, in handle order.
    let mut bindable: Vec<(u64, BufferMeta)> = store
        .iter()
        .map(|(id, array)| (id, array.meta.clone()))
        .collect();
    bindable.sort_by_key(|(id, _)| *id);

    for (entity, id, object, kind, visibility, fields, children, drawn_by, parent) in objects {
        let child_objects: Vec<Entity> = children
            .into_iter()
            .flatten()
            .copied()
            .filter(|child| objects.contains(*child))
            .collect();

        let drawn: Vec<ActorRow> = drawn_by
            .into_iter()
            .flat_map(|list| list.iter())
            .filter_map(|entity| {
                let (actor_id, kind, params, colour, subset) = actors.get(entity).ok()?;
                // A kind with no registration cannot be drawn or configured, so
                // there is nothing useful to show for it.
                let registered = registry.get(kind.0)?;
                Some(ActorRow {
                    entity,
                    id: actor_id.0,
                    label: registered.label,
                    specs: registered.params,
                    params: params.0.clone(),
                    colour: colour.clone(),
                    subset: subset.clone(),
                })
            })
            .collect();

        let available: Vec<(&'static str, &'static str)> = registry
            .for_dataset(*kind)
            .map(|kind| (kind.id, kind.label))
            .collect();

        for array in &object.arrays {
            owners.insert(
                array.handle.id(),
                Owner {
                    object: id.0,
                    name: array.meta.name.clone(),
                    fields: Vec::new(),
                },
            );
        }
        // A field names bytes that were already uploaded as a buffer, so this
        // usually annotates an entry that exists. Grids are the exception:
        // their fields are the only thing uploaded, so the entry starts here.
        if let Some(fields) = fields {
            for (name, field) in &fields.0 {
                owners
                    .entry(field.array.id())
                    .or_insert_with(|| Owner {
                        object: id.0,
                        name: name.clone(),
                        fields: Vec::new(),
                    })
                    .fields
                    .push(name.clone());
            }
        }
        // An actor's selection is an array too, and it is held by the actor
        // rather than the object. Without this the inventory calls it
        // unreferenced, which is exactly backwards.
        for actor in &drawn {
            if let Subset::Selected { array, .. } = &actor.subset {
                owners.insert(
                    array.id(),
                    Owner {
                        object: id.0,
                        name: "subset".into(),
                        fields: Vec::new(),
                    },
                );
            }
        }

        let mut field_names: Vec<FieldRow> = fields
            .map(|fields| {
                fields
                    .0
                    .iter()
                    .map(|(name, field)| FieldRow {
                        name: name.clone(),
                        kind: match field.kind {
                            FieldKind::Scalar => "scalar",
                            FieldKind::Vector => "vector",
                            FieldKind::Tensor(_) => "tensor",
                        },
                    })
                    .collect()
            })
            .unwrap_or_default();
        field_names.sort_by(|a, b| a.name.cmp(&b.name));

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
                kind: *kind,
                visible: *visibility != Visibility::Hidden,
                arrays: object.arrays.len(),
                bytes: object.total_bytes(),
                fields: field_names,
                actors: drawn,
                available,
                children: child_objects,
            },
        );
    }

    for owner in owners.values_mut() {
        owner.fields.sort();
    }

    let handle = |entity: &Entity| rows.get(entity).map(|row| row.id).unwrap_or(u64::MAX);
    roots.sort_by_key(handle);
    let mut ordered: Vec<Entity> = rows.keys().copied().collect();
    ordered.sort_by_key(handle);

    // Objects and loose arrays both, or the count and the size would disagree:
    // the listing shows every array in memory, so the total has to cover them.
    let object_bytes: u64 = rows.values().map(|row| row.bytes).sum();
    let held_bytes: u64 = store
        .iter()
        .filter_map(|(_, array)| array.meta.byte_length())
        .sum();
    let total_bytes = object_bytes + held_bytes;

    Gathered {
        rows,
        roots,
        ordered,
        owners,
        held,
        bindable,
        total_bytes,
    }
}
