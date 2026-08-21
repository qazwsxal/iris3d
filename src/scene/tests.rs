//! Tests for the scene tree and the commands that change it.

use super::registry::ActorRegistry;
use super::*;
use crate::counter::{GlobalIDCounter, UniqueID};
use crate::model::SceneError;
use crate::redraw::KeepAwake;
use bevy::transform::TransformPlugin;
use tokio::sync::oneshot;

fn app() -> App {
    let mut app = App::new();
    // Enough of the world for the drain to run: the assets it ingests into,
    // the handle counter, an empty registry (nothing here asks to be drawn)
    // and the channel it reads. `TransformPlugin` is what makes
    // `GlobalTransform` mean anything, which the reparent path depends on.
    app.add_plugins(TransformPlugin);
    app.add_message::<AssetEvent<DataArray>>();
    app.init_resource::<Assets<DataArray>>();
    // Geometry is an asset like any other, and `mark_dirty` watches it: a
    // filter rewriting a mesh has to reach the actors drawing it.
    app.add_message::<AssetEvent<Mesh>>();
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<DataStore>();
    app.init_resource::<GlobalIDCounter>();
    app.init_resource::<ActorRegistry>();
    // No pathway is added here, so nothing would otherwise name one, and the
    // messages that quote the backend would read "the no backend has...".
    app.world_mut()
        .resource_mut::<ActorRegistry>()
        .served_by("test");
    app.init_resource::<KeepAwake>();
    app.init_resource::<CommandBus>();
    // Placements follow the parents the drain writes, so the whole chain
    // has to run for a listing to reflect what is on screen.
    app.add_systems(
        Update,
        (
            apply_scene_commands,
            link::sync_placements,
            link::apply_shown,
        )
            .chain(),
    );
    app
}

/// Queues a command the way the gRPC side does, and hands back the reply.
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

fn create(app: &App, name: &str) -> oneshot::Receiver<ObjectSummary> {
    send(app, |reply| SceneCommand::CreateObject {
        name: name.into(),
        reply,
    })
}

fn place(app: &App, id: u64, x: f32) -> oneshot::Receiver<Result<(), SceneError>> {
    send(app, |reply| SceneCommand::SetTransform {
        id,
        translation: Some(Vec3::new(x, 0.0, 0.0)),
        rotation: None,
        scale: None,
        reply,
    })
}

fn reparent(app: &App, id: u64, parent: Option<u64>) -> oneshot::Receiver<Result<(), SceneError>> {
    send(app, |reply| SceneCommand::SetParent {
        id,
        parent,
        keep_world_transform: true,
        reply,
    })
}

/// A kind that draws nothing and needs nothing, so a test can make actors
/// without a rendering backend compiled in.
fn marker(app: &mut App) {
    app.world_mut()
        .resource_mut::<ActorRegistry>()
        .register(registry::ActorKind {
            id: "marker",
            label: "Marker",
            params: &[],
            apply: |_, _| {},
        });
}

fn two_objects(app: &mut App) -> (u64, u64) {
    let mut first = create(app, "first");
    let mut second = create(app, "second");
    app.update();
    (
        first.try_recv().expect("a reply").id,
        second.try_recv().expect("a reply").id,
    )
}

fn add(app: &mut App, parents: Vec<u64>) -> u64 {
    let mut added = send(app, |reply| SceneCommand::AddActor {
        parents,
        kind: "marker".into(),
        params: crate::model::ParamMap::default(),
        reply,
    });
    app.update();
    added.try_recv().expect("a reply").expect("an actor").id
}

/// How many copies of anything are on screen.
fn placements(app: &mut App) -> usize {
    app.world_mut()
        .query::<&Placement>()
        .iter(app.world())
        .len()
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

/// The same, of a shape the caller chooses, for tests about what an input
/// will accept.
fn shaped(name: &str, shape: &[u64]) -> NamedBuffer {
    let elements: u64 = shape.iter().product();
    NamedBuffer {
        meta: BufferMeta {
            name: name.into(),
            dtype: Dtype::Uint8,
            shape: shape.to_vec(),
        },
        data: vec![0; elements as usize],
        strings: Vec::new(),
    }
}

/// Arrays arrive on their own: a handle each, no object, no actor. Data used
/// to be reachable only by making an object out of it, which conflated
/// "hold these numbers" with "put a node in the tree".
#[test]
fn uploaded_arrays_are_held_without_an_object() {
    let mut app = app();
    let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
        arrays: vec![array("xyz", 12), array("t", 4)],
        reply,
    });
    app.update();

    let summaries = uploaded.try_recv().expect("a reply");
    let handles: Vec<u64> = summaries.iter().map(|array| array.id).collect();
    assert_eq!(summaries.len(), 2);
    assert_eq!(
        summaries.iter().map(|a| a.meta.name()).collect::<Vec<_>>(),
        ["xyz", "t"],
        "handles come back in declaration order"
    );
    assert_eq!(
        app.world().resource::<DataStore>().iter().count(),
        2,
        "the store keeps the bytes alive"
    );
    let objects = app
        .world_mut()
        .query::<&SceneObject>()
        .iter(app.world())
        .count();
    assert_eq!(objects, 0, "an upload of arrays creates no object");

    // Forgetting reports what was actually held, so a caller learns which
    // of its handles it was wrong about.
    let mut released = send(&app, |reply| SceneCommand::ReleaseData {
        ids: vec![handles[0], 9_999],
        reply,
    });
    app.update();
    assert_eq!(released.try_recv().expect("a reply"), vec![handles[0]]);
    assert_eq!(app.world().resource::<DataStore>().iter().count(), 1);
}

/// The ordinary case: both objects are really in the world, so the child's
/// local transform absorbs the parent's and it does not move.
#[test]
fn keeping_the_world_transform_offsets_the_child() {
    let mut app = app();
    let mut parent = create(&app, "parent");
    let mut child = create(&app, "child");
    app.update();
    let parent = parent.try_recv().expect("a reply").id;
    let child = child.try_recv().expect("a reply").id;

    place(&app, parent, 5.0);
    place(&app, child, 1.0);
    app.update();

    let mut moved = reparent(&app, child, Some(parent));
    app.update();
    assert_eq!(moved.try_recv().expect("a reply"), Ok(()));

    let entity = app
        .world_mut()
        .query::<(Entity, &UniqueID)>()
        .iter(app.world())
        .find(|(_, id)| id.0 == child)
        .map(|(entity, _)| entity)
        .expect("the child is in the world");
    let local = app.world().get::<Transform>(entity).expect("a transform");
    assert_eq!(
        local.translation,
        Vec3::new(-4.0, 0.0, 0.0),
        "the child should stay where it was in world space"
    );
}

/// The same command against an object created earlier in the *same* drain.
/// Its handle is already in `index`, but `Commands` has not spawned it, so
/// neither world transform exists. Answering with the origin would misplace
/// the object without saying so, so the command has to fail instead.
///
/// No client can reach this today — naming the object needs the handle, and
/// the handle only arrives in a reply, by which point the spawn has landed.
/// The test builds the batch by hand, since handles are allocated in order.
#[test]
fn refuses_to_keep_the_world_transform_of_an_object_made_this_tick() {
    let mut app = app();
    let mut parent = create(&app, "parent");
    app.update();
    let parent = parent.try_recv().expect("a reply").id;

    // Both in one batch, the second naming what the first will create.
    let mut child = create(&app, "child");
    let mut moved = reparent(&app, parent + 1, Some(parent));
    app.update();

    assert_eq!(
        child.try_recv().expect("a reply").id,
        parent + 1,
        "handles come out of one sequence, so the batch could name it"
    );
    assert_eq!(
        moved.try_recv().expect("a reply"),
        Err(SceneError::NoSuchObject(parent + 1)),
        "a silent misplacement is worse than a refusal"
    );
}

/// An actor with no parent named gets an object made for it, named after
/// its kind.
///
/// An actor has no place of its own, so it has to end up under something.
/// Refusing instead would make `CreateObject` the opening line of every
/// drawing a client does, to make a node it has no other use for.
#[test]
fn an_actor_with_no_parent_is_given_an_object() {
    let mut app = app();
    marker(&mut app);

    let mut added = send(&app, |reply| SceneCommand::AddActor {
        parents: vec![],
        kind: "marker".into(),
        params: crate::model::ParamMap::default(),
        reply,
    });
    app.update();
    let actor = added.try_recv().expect("a reply").expect("an actor");

    let mut listed = send(&app, |reply| SceneCommand::ListObjects { reply });
    app.update();
    let objects = listed.try_recv().expect("a reply");
    assert_eq!(
        objects
            .iter()
            .map(|o| (o.id, o.name.as_str()))
            .collect::<Vec<_>>(),
        vec![(actor.parents[0], "Marker")],
        "exactly one object, named after the kind, holding the actor"
    );
}

/// A refused actor leaves no object behind.
///
/// The object is created last, after every check. Creating it up front
/// would litter the scene with empty nodes whenever a client got a binding
/// or a kind name wrong.
#[test]
fn an_actor_that_cannot_be_added_creates_nothing() {
    let mut app = app();

    let mut added = send(&app, |reply| SceneCommand::AddActor {
        parents: vec![],
        kind: "no-such-kind".into(),
        params: crate::model::ParamMap::default(),
        reply,
    });
    app.update();
    let refusal = added.try_recv().expect("a reply").err();
    assert_eq!(
        refusal,
        Some(SceneError::UnknownKind {
            kind: "no-such-kind".into(),
            backend: "test",
        })
    );
    // The pathway belongs in the message, not only in the variant. A kind
    // can be perfectly real under another backend, and a client that cannot
    // tell "you mistyped it" from "you are on the wrong pathway" has to
    // guess.
    let said = refusal.expect("a refusal").to_string();
    assert!(said.contains("test backend"), "{said}");

    let mut listed = send(&app, |reply| SceneCommand::ListObjects { reply });
    app.update();
    assert!(
        listed.try_recv().expect("a reply").is_empty(),
        "a refusal must not leave an empty object behind"
    );
}

/// A rebind to an array the input cannot read is refused, and the actor
/// keeps the array it had.
///
/// `sanitise` alone is not enough here: for an array input it only confirms
/// that a handle is a handle. A rebind has to be checked against the input's
/// declared element type and shape as well, or the declaration would hold
/// when the actor is added and stop holding the moment it is changed — which
/// is the call a client uses to rebind geometry.
#[test]
fn an_actor_keeps_its_binding_when_a_rebind_is_refused() {
    const INPUTS: &[crate::model::ParamSpec] = &[crate::model::ParamSpec {
        id: "field",
        label: "Field",
        kind: crate::model::ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: true,
            structural: true,
        },
    }];

    let mut app = app();
    app.world_mut()
        .resource_mut::<ActorRegistry>()
        .register(registry::ActorKind {
            id: "bound",
            label: "Bound",
            params: INPUTS,
            apply: |_, _| {},
        });

    let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
        arrays: vec![
            array("first", 4),
            shaped("wrong", &[2, 3]),
            array("second", 8),
        ],
        reply,
    });
    app.update();
    let handles: Vec<u64> = uploaded
        .try_recv()
        .expect("a reply")
        .iter()
        .map(|array| array.id)
        .collect();
    let (first, wrong, second) = (handles[0], handles[1], handles[2]);

    let bind = |handle: u64| {
        let mut params = crate::model::ParamMap::default();
        params.insert("field".into(), crate::model::ParamValue::Data(handle));
        params
    };

    let mut added = send(&app, |reply| SceneCommand::AddActor {
        parents: vec![],
        kind: "bound".into(),
        params: bind(first),
        reply,
    });
    app.update();
    let actor = added.try_recv().expect("a reply").expect("an actor").id;

    let mut refused = send(&app, |reply| SceneCommand::SetActor {
        id: actor,
        params: bind(wrong),
        visible: None,
        parents: None,
        reply,
    });
    app.update();
    let refusal = refused
        .try_recv()
        .expect("a reply")
        .expect_err("a [2, 3] array does not fit an [n] input");
    let SceneError::BadBinding {
        ref kind,
        input,
        ref reason,
    } = refusal
    else {
        panic!("expected a refused binding, got {refusal:?}");
    };
    assert_eq!((kind.as_str(), input), ("bound", "field"));
    // The reason reaches the client that got it wrong, as it does from
    // `AddActor`.
    assert!(reason.contains("[2, 3]"), "{reason}");
    assert!(reason.contains("[n]"), "{reason}");

    let bound = |app: &mut App| {
        let mut listed = send(app, |reply| SceneCommand::ListActors {
            parent: None,
            reply,
        });
        app.update();
        let listing = listed.try_recv().expect("a reply").expect("a listing");
        crate::model::data(&listing[0].params, "field")
    };
    assert_eq!(
        bound(&mut app),
        Some(first),
        "a refused rebind must leave the actor exactly as it was"
    );

    // And a binding that does fit still goes through: the check refuses
    // what the input cannot read, not every change.
    let mut accepted = send(&app, |reply| SceneCommand::SetActor {
        id: actor,
        params: bind(second),
        visible: None,
        parents: None,
        reply,
    });
    app.update();
    accepted.try_recv().expect("a reply").expect("an actor");
    assert_eq!(bound(&mut app), Some(second));
}

/// An optional input can be let go of, and a required one cannot.
///
/// The merge rule makes this need a value of its own: a partial map leaves
/// anything absent alone, so absence already means "unchanged" and cannot
/// also mean "clear". Without `Unset` an optional input could be bound once
/// and never released, by any client.
#[test]
fn an_optional_input_can_be_unbound_and_a_required_one_cannot() {
    const INPUTS: &[crate::model::ParamSpec] = &[
        crate::model::ParamSpec {
            id: "field",
            label: "Field",
            kind: crate::model::ParamKind::Array {
                dtypes: &[Dtype::Uint8],
                shape: &[0],
                required: true,
                structural: true,
            },
        },
        crate::model::ParamSpec {
            id: "mask",
            label: "Mask",
            kind: crate::model::ParamKind::Array {
                dtypes: &[Dtype::Uint8],
                shape: &[0],
                required: false,
                structural: true,
            },
        },
    ];

    let mut app = app();
    app.world_mut()
        .resource_mut::<ActorRegistry>()
        .register(registry::ActorKind {
            id: "maskable",
            label: "Maskable",
            params: INPUTS,
            apply: |_, _| {},
        });

    let mut uploaded = send(&app, |reply| SceneCommand::UploadData {
        arrays: vec![array("field", 4), array("mask", 4)],
        reply,
    });
    app.update();
    let handles: Vec<u64> = uploaded
        .try_recv()
        .expect("a reply")
        .iter()
        .map(|array| array.id)
        .collect();

    let mut params = crate::model::ParamMap::default();
    params.insert("field".into(), crate::model::ParamValue::Data(handles[0]));
    params.insert("mask".into(), crate::model::ParamValue::Data(handles[1]));
    let mut added = send(&app, |reply| SceneCommand::AddActor {
        parents: vec![],
        kind: "maskable".into(),
        params,
        reply,
    });
    app.update();
    let actor = added.try_recv().expect("a reply").expect("an actor").id;

    let clear = |input: &str| {
        let mut params = crate::model::ParamMap::default();
        params.insert(input.into(), crate::model::ParamValue::Unset);
        params
    };
    let bound = |app: &mut App, input: &'static str| {
        let mut listed = send(app, |reply| SceneCommand::ListActors {
            parent: None,
            reply,
        });
        app.update();
        let listing = listed.try_recv().expect("a reply").expect("a listing");
        crate::model::data(&listing[0].params, input)
    };

    let mut cleared = send(&app, |reply| SceneCommand::SetActor {
        id: actor,
        params: clear("mask"),
        visible: None,
        parents: None,
        reply,
    });
    app.update();
    cleared.try_recv().expect("a reply").expect("an actor");
    assert_eq!(bound(&mut app, "mask"), None, "the mask should be let go");
    assert_eq!(
        bound(&mut app, "field"),
        Some(handles[0]),
        "clearing one input must leave the others alone"
    );

    // The required one is refused by the same gate that refuses never
    // having bound it, and says the same thing.
    let mut refused = send(&app, |reply| SceneCommand::SetActor {
        id: actor,
        params: clear("field"),
        visible: None,
        parents: None,
        reply,
    });
    app.update();
    let refusal = refused
        .try_recv()
        .expect("a reply")
        .expect_err("a required input cannot be cleared");
    assert!(
        matches!(refusal, SceneError::MissingInput { input, .. } if input == "field"),
        "expected a missing input, got {refusal:?}"
    );
    assert_eq!(
        bound(&mut app, "field"),
        Some(handles[0]),
        "a refused clear must leave the actor exactly as it was"
    );
}

/// Every kind the backend registered reaches a client, with the label and
/// the parameters it declared.
///
/// This is the only way a script learns what the running build can draw —
/// there is no table on the client side to fall back on — so a kind that
/// registers but does not list is invisible.
#[test]
fn listing_kinds_reports_every_registered_kind() {
    const PARAMS: &[crate::model::ParamSpec] = &[crate::model::ParamSpec {
        id: "size",
        label: "size",
        kind: crate::model::ParamKind::Float {
            default: 1.0,
            min: 0.0,
            max: 2.0,
            logarithmic: false,
        },
    }];

    let mut app = app();
    marker(&mut app);
    app.world_mut()
        .resource_mut::<ActorRegistry>()
        .register(registry::ActorKind {
            id: "sized",
            label: "Sized",
            params: PARAMS,
            apply: |_, _| {},
        });

    let mut listed = send(&app, |reply| SceneCommand::ListActorKinds { reply });
    app.update();
    let kinds = listed.try_recv().expect("a reply");

    let ids: Vec<&str> = kinds.iter().map(|kind| kind.id.as_str()).collect();
    assert_eq!(ids, vec!["marker", "sized"], "registration order");

    let sized = kinds
        .iter()
        .find(|kind| kind.id == "sized")
        .expect("listed");
    assert_eq!(sized.label, "Sized");
    assert_eq!(sized.params.len(), 1);
    assert_eq!(sized.params[0].id, "size");
}

/// One actor under two objects is drawn twice and stays one actor.
///
/// The whole point of splitting an actor from its placements. Two actors
/// binding the same arrays would look the same on screen and then have to
/// be configured one at a time; this is one drawing, one mesh, and one
/// thing to change.
#[test]
fn one_actor_is_drawn_under_every_object_it_names() {
    let mut app = app();
    marker(&mut app);

    let (first, second) = two_objects(&mut app);
    let actor = add(&mut app, vec![first, second]);

    let mut listed = send(&app, |reply| SceneCommand::ListActors {
        parent: None,
        reply,
    });
    app.update();
    let listing = listed.try_recv().expect("a reply").expect("a listing");
    assert_eq!(
        listing.iter().map(|a| a.id).collect::<Vec<_>>(),
        vec![actor],
        "two placements are still one actor"
    );
    assert_eq!(listing[0].parents, vec![first, second]);

    // And each object reports it, because each of them draws it.
    let mut objects = send(&app, |reply| SceneCommand::ListObjects { reply });
    app.update();
    assert_eq!(
        objects
            .try_recv()
            .expect("a reply")
            .iter()
            .map(|o| (o.id, o.actors.iter().map(|a| a.id).collect::<Vec<_>>()))
            .collect::<Vec<_>>(),
        vec![(first, vec![actor]), (second, vec![actor])]
    );
}

/// Deleting an object costs an actor that one appearance and nothing else.
///
/// The actor is not in the tree — only its placements are — so a deletion
/// reaches exactly one of them. Destroying the actor with the object would
/// mean tidying up one of the three places a drawing appears destroys the
/// drawing.
#[test]
fn deleting_one_object_leaves_the_actor_drawn_under_the_others() {
    let mut app = app();
    marker(&mut app);

    let (first, second) = two_objects(&mut app);
    let actor = add(&mut app, vec![first, second]);

    let mut deleted = send(&app, |reply| SceneCommand::DeleteObject {
        id: first,
        reply,
    });
    app.update();
    assert_eq!(
        deleted.try_recv().expect("a reply").objects,
        vec![first],
        "a deletion takes the object named and nothing else"
    );

    let mut listed = send(&app, |reply| SceneCommand::ListActors {
        parent: None,
        reply,
    });
    app.update();
    let listing = listed.try_recv().expect("a reply").expect("a listing");
    assert_eq!(
        listing
            .iter()
            .map(|a| (a.id, a.parents.clone()))
            .collect::<Vec<_>>(),
        vec![(actor, vec![second])],
        "the actor survives, drawn under what is left"
    );
    assert!(
        listing[0].visible,
        "and its own visibility setting is untouched"
    );
}

/// An actor under nothing draws nothing, and comes back when placed.
///
/// No hidden node arranges this and nothing marks the actor: with the mesh
/// on the placements, having none *is* being off screen.
#[test]
fn an_actor_with_no_parents_has_no_placements() {
    let mut app = app();
    marker(&mut app);

    let (first, second) = two_objects(&mut app);
    let actor = add(&mut app, vec![first]);

    let set = |app: &App, parents: Vec<u64>| {
        send(app, move |reply| SceneCommand::SetActor {
            id: actor,
            params: crate::model::ParamMap::default(),
            visible: None,
            parents: Some(parents),
            reply,
        })
    };

    let mut cleared = set(&app, vec![]);
    app.update();
    assert!(
        cleared
            .try_recv()
            .expect("a reply")
            .expect("an actor")
            .parents
            .is_empty()
    );
    assert_eq!(placements(&mut app), 0, "nothing is drawn");

    let mut placed = set(&app, vec![second]);
    app.update();
    assert_eq!(
        placed
            .try_recv()
            .expect("a reply")
            .expect("an actor")
            .parents,
        vec![second],
        "and it draws again once it has somewhere to be"
    );
    assert_eq!(placements(&mut app), 1);
}
