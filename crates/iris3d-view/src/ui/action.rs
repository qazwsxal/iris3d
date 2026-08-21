//! What the interface can do to the scene, and the doing of it.
//!
//! The UI never mutates directly. It reads the world, pushes [`UiAction`]s onto
//! a queue, and [`apply_actions`] drains that queue after drawing has finished.
//!
//! Two reasons. egui closures would otherwise need several conflicting mutable
//! borrows at once; and keeping every mutation in one place makes the list of
//! what the interface *can* do readable in one screen. A tab gets `&Gathered`
//! and a queue to push onto, and nothing more.

use super::*;

/// Something the user asked for, applied after the UI has finished drawing.
pub enum UiAction {
    Select(Entity),
    /// Show an actor's controls. The second entity is the object whose row it
    /// was clicked in — an actor drawn under three objects appears in three
    /// rows, so which one was clicked is the only thing that says which object
    /// to highlight. `None` for one that is drawn nowhere.
    SelectActor(Entity, Option<Entity>),
    SelectArray(AssetId<DataArray>),
    ToggleVisibility(Entity),
    Delete(u64),
    Frame(Entity),
    FrameAll,
    /// Start drawing an object an additional way. The object keeps the actors it
    /// already has — this adds, it does not replace.
    ///
    /// Opens a draft rather than spawning: every kind has at least one required
    /// input, so there is nothing useful to create before they are chosen.
    AddActor(Entity, &'static str),
    RemoveActor(Entity),
    /// Change one parameter of an actor, leaving the rest alone.
    SetParam(Entity, &'static str, ParamValue),

    /// Draw an actor under one more object, or stop drawing it under one.
    ///
    /// The whole set is replaced rather than added to, because that is what
    /// `SetActor` takes — an actor's parents are a set, not a list of edges, and
    /// sending the set it should end up with is what makes adding and removing
    /// the same operation.
    SetActorParents(u64, Vec<u64>),
    /// Nest an object inside another, or `None` to make it a root.
    SetObjectParent(u64, Option<u64>),

    SelectFilter(u64),
    /// Change one parameter of a filter, leaving the rest alone.
    ///
    /// By handle rather than entity, because it goes out as a `SceneCommand` and
    /// that is the name a command speaks in.
    SetFilterParam(u64, &'static str, ParamValue),
    RemoveFilter(u64),
    /// Open the create form for a kind, optionally remembering what to wire the
    /// result into.
    OfferFilter {
        kind: &'static str,
        then: Option<(&'static str, Target)>,
    },
    /// Build whatever is drafted for real.
    Create {
        kind: &'static str,
        params: ParamMap,
        making: Making,
    },
    /// Change one parameter of the draft. Goes through the queue like every
    /// other edit rather than being written where it is read, so there is still
    /// one place that says what the interface can do.
    SetDraftParam(&'static str, ParamValue),
    CancelDraft,
}

#[derive(Resource, Default)]
pub struct PendingActions(pub Vec<UiAction>);

/// Replies still in the air, and the last thing that was refused.
///
/// Every other action here is a fire-and-forget: `DeleteObject` drops its reply
/// channel because there is nothing to learn from it. Creating a filter is not
/// like that. The reply carries the handles its outputs were allocated, and
/// those are the whole point — without them the interface cannot wire the new
/// filter to the thing it was created for, and would have to make the user do it
/// by hand immediately after offering not to.
///
/// So the receiver is kept and polled. It also gives refusals somewhere to land:
/// a binding the backend will not accept, or a cycle, otherwise fails in silence
/// and looks like a button that does nothing.
#[derive(Resource, Default)]
pub(super) struct Pending {
    pub(super) waiting: Vec<Job>,
    /// Shown at the foot of the panel until the next thing succeeds.
    pub(super) error: Option<String>,
}

pub(super) struct Job {
    pub(super) reply: tokio::sync::oneshot::Receiver<Result<FilterSummary, SceneError>>,
    /// Which output to bind, and where.
    pub(super) then: Option<(&'static str, Target)>,
    /// Show the filter this reply is about once it lands.
    ///
    /// True for a creation and false for an edit. Without the distinction a
    /// slider drag — which is one `SetFilter` per frame — would drag the
    /// selection back to the filter being edited every frame, and take it off
    /// whatever the user clicked in the middle of the drag.
    pub(super) select: bool,
}

/// Applies what the UI asked for.
///
/// Deletion goes through [`SceneCommand`] rather than despawning directly, so
/// the UI takes exactly the same path a scripted client does — including the
/// detaching of child objects rather than destroying them.
#[allow(clippy::too_many_arguments)]
/// Turns a click in the 3D view into the same action a tree click emits.
///
/// The viewport reports *what was hit* and nothing about selection: what being
/// selected means belongs here, and picking should not stop working because the
/// interface was not built. So this is the one place the two meet, and it goes
/// through `UiAction` like everything else rather than writing `UiState`
/// directly — an interaction that bypassed the queue would be the one the rest
/// of the interface could not account for.
///
/// Ignored while egui owns the mouse. A click on a panel that happens to lie
/// over geometry is a click on the panel.
pub(super) fn take_picks(
    mut picks: MessageReader<iris3d_scene::Picked>,
    captured: Res<PointerCaptured>,
    mut actions: ResMut<PendingActions>,
) {
    for pick in picks.read() {
        if captured.0 {
            continue;
        }
        actions
            .0
            .push(UiAction::SelectActor(pick.actor, Some(pick.object)));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_actions(
    mut commands: Commands,
    mut actions: ResMut<PendingActions>,
    mut state: ResMut<UiState>,
    mut selection: ResMut<Selection>,
    mut frame: ResMut<FrameRequest>,
    registry: Res<ActorRegistry>,
    filter_registry: Res<iris3d_filter::FilterRegistry>,
    mut visibility: Query<&mut Visibility>,
    mut params: Query<(&ActorKindId, &mut ActorParams)>,
    actor_entities: Query<(), With<ActorKindId>>,
    // The handle as well as the membership: an actor is asked for by object
    // handle, because that is the name the command speaks in.
    scene_objects: Query<&iris3d_core::counter::UniqueID, With<iris3d_scene::SceneObject>>,
    bus: Res<iris3d_scene::CommandBus>,
    filters: Res<FilterBus>,
    mut pending: ResMut<Pending>,
) {
    for action in actions.0.drain(..) {
        match action {
            UiAction::Select(entity) => {
                selection.object = if selection.object == Some(entity) {
                    None
                } else {
                    Some(entity)
                };
                // The Actors tab would otherwise keep showing the controls for
                // an actor of whatever was selected before, tinted under a
                // group that is no longer highlighted.
                selection.actor = None;
            }
            UiAction::SelectActor(entity, under) => {
                selection.actor = Some(entity);
                // Select the object it was clicked under too: the outline, the
                // tree highlight and the tint in the actor list all key off the
                // object selection, and three of them disagreeing is worse than
                // the outline moving. An actor drawn nowhere clears it rather
                // than leaving it on whatever was picked before.
                selection.object = under.filter(|parent| scene_objects.contains(*parent));
            }
            UiAction::SelectArray(id) => selection.array = Some(id),
            UiAction::ToggleVisibility(entity) => {
                if let Ok(mut visibility) = visibility.get_mut(entity) {
                    *visibility = match *visibility {
                        Visibility::Hidden => Visibility::Inherited,
                        _ => Visibility::Hidden,
                    };
                }
            }
            UiAction::Delete(id) => {
                let (reply, _) = tokio::sync::oneshot::channel();
                let _ = bus.sender().send(SceneCommand::DeleteObject { id, reply });
            }
            UiAction::Frame(entity) => frame.0 = Some(FrameTarget::Subtree(entity)),
            UiAction::FrameAll => frame.0 = Some(FrameTarget::All),

            // Placed under the selected object. What it *draws* is bound
            // afterwards, through the input pickers.
            UiAction::AddActor(object, kind) => {
                let Some(registered) = registry.get(kind) else {
                    continue;
                };
                state.draft = Some(Draft {
                    kind: registered.id,
                    params: registered.defaults(),
                    making: Making::Actor(object),
                });
            }
            UiAction::RemoveActor(entity) => {
                // Every copy of it goes: `Placements` is a linked relationship,
                // so despawning the actor takes its placements with it.
                if actor_entities.contains(entity) {
                    commands.entity(entity).despawn();
                    if selection.actor == Some(entity) {
                        selection.actor = None;
                    }
                }
            }

            // Each of these mutates a component the draw backends watch, so
            // `mark_dirty` picks them up and the geometry rebuilds. Nothing
            // here touches meshes directly.
            UiAction::SetParam(entity, id, value) => {
                let Ok((kind, mut current)) = params.get_mut(entity) else {
                    continue;
                };
                // Through the same sanitiser a client's value goes through, so
                // there is one definition of what a parameter may be.
                let Some(value) = registry
                    .get(kind.0)
                    .and_then(|kind| kind.spec(id))
                    .and_then(|spec| spec.kind.sanitise(value))
                else {
                    continue;
                };
                current.0.insert(id.to_string(), value);
            }

            UiAction::SetActorParents(id, parents) => {
                let (reply, _) = tokio::sync::oneshot::channel();
                let _ = bus.sender().send(SceneCommand::SetActor {
                    id,
                    params: ParamMap::default(),
                    visible: None,
                    parents: Some(parents),
                    reply,
                });
            }
            UiAction::SetObjectParent(id, parent) => {
                let (reply, _) = tokio::sync::oneshot::channel();
                let _ = bus.sender().send(SceneCommand::SetParent {
                    id,
                    parent,
                    // The object stays where it looks like it is. Re-parenting
                    // by dragging a wire says nothing about wanting to move it,
                    // and having it jump would be a surprise.
                    keep_world_transform: true,
                    reply,
                });
            }

            UiAction::SelectFilter(id) => selection.filter = Some(id),

            // Filters go down the command channel rather than being written
            // here. Not for consistency's sake: `set` merges the change over the
            // current map, re-checks every binding and refuses a cycle, and
            // duplicating that in the interface would be a second, worse copy of
            // the rules.
            UiAction::SetFilterParam(id, input, value) => {
                let mut params = ParamMap::default();
                params.insert(input.to_string(), value);
                let (reply, receiver) = tokio::sync::oneshot::channel();
                if filters
                    .sender()
                    .send(FilterCommand::Set { id, params, reply })
                    .is_ok()
                {
                    pending.waiting.push(Job {
                        reply: receiver,
                        then: None,
                        select: false,
                    });
                }
            }
            UiAction::RemoveFilter(id) => {
                let (reply, _) = tokio::sync::oneshot::channel();
                let _ = filters.sender().send(FilterCommand::Remove { id, reply });
                selection.filter = None;
            }

            UiAction::OfferFilter { kind, then } => {
                let Some(registered) = filter_registry.get(kind) else {
                    continue;
                };
                // Settings at their defaults, inputs left empty — `normalise`
                // over an empty map is exactly that, and is what a filter kind
                // has in place of the `defaults()` an actor kind carries.
                state.draft = Some(Draft {
                    kind: registered.id,
                    params: registered.normalise(&ParamMap::default()),
                    making: Making::Filter(then),
                });
                state.tab = Tab::Filters;
            }
            UiAction::SetDraftParam(input, value) => {
                let Some(draft) = &mut state.draft else {
                    continue;
                };
                // Whichever registry owns the kind being drafted. The two
                // declare parameters the same way, which is what lets one form
                // serve both, but they are separate namespaces.
                let spec = match draft.making {
                    Making::Filter(_) => filter_registry
                        .get(draft.kind)
                        .and_then(|kind| kind.params.iter().find(|spec| spec.id == input)),
                    Making::Actor(_) => registry
                        .get(draft.kind)
                        .and_then(|kind| kind.params.iter().find(|spec| spec.id == input)),
                };
                let Some(value) = spec.and_then(|spec| spec.kind.sanitise(value)) else {
                    continue;
                };
                draft.params.insert(input.to_string(), value);
            }
            UiAction::CancelDraft => state.draft = None,
            UiAction::Create {
                kind,
                params,
                making,
            } => {
                match making {
                    Making::Filter(then) => {
                        let (reply, receiver) = tokio::sync::oneshot::channel();
                        if filters
                            .sender()
                            .send(FilterCommand::Add {
                                kind: kind.to_string(),
                                params,
                                reply,
                            })
                            .is_ok()
                        {
                            pending.waiting.push(Job {
                                reply: receiver,
                                then,
                                select: true,
                            });
                        }
                    }
                    // Down the command channel like everything else, so the
                    // binding check that refuses a scripted `add_actor` refuses
                    // this too. The reply is dropped: an actor's handles are not
                    // needed afterwards the way a filter's outputs are.
                    Making::Actor(object) => {
                        let (reply, _) = tokio::sync::oneshot::channel();
                        let _ = bus.sender().send(SceneCommand::AddActor {
                            // One object. Drawing it somewhere else as well is a
                            // second parent, which a client asks for by handle —
                            // there is nothing to click on to mean "and also
                            // there".
                            parents: scene_objects
                                .get(object)
                                .map(|id| vec![id.0])
                                .unwrap_or_default(),
                            kind: kind.to_string(),
                            params,
                            // Selections are computed by a client, not clicked
                            // together here, so the tree only ever adds a
                            // whole-dataset one.
                            reply,
                        });
                    }
                }
                // Closed on asking rather than on landing. The reply is a tick
                // away at least, and leaving the form up meanwhile invites a
                // second click that would build a second one.
                state.draft = None;
            }
        }
    }
}

/// Picks up the replies to the filter commands the interface sent.
///
/// Runs before [`apply_actions`] and pushes onto the same queue, so a filter
/// created for an actor's empty input is bound to it in the tick its handles
/// arrive, not the one after.
pub(super) fn collect_replies(mut pending: ResMut<Pending>, mut actions: ResMut<PendingActions>) {
    let mut still_waiting = Vec::new();
    for mut job in std::mem::take(&mut pending.waiting) {
        match job.reply.try_recv() {
            Ok(Ok(summary)) => {
                pending.error = None;
                // Show what was just made, without a second click into the list.
                if job.select {
                    actions.0.push(UiAction::SelectFilter(summary.id));
                }
                let Some((output, target)) = job.then else {
                    continue;
                };
                // The handles are the reason the reply was kept at all.
                let Some((_, handle)) = summary.outputs.iter().find(|(id, _)| id == output) else {
                    warn!("ui: filter {} produced no output \"{output}\"", summary.id);
                    continue;
                };
                actions.0.push(match target {
                    Target::Actor(entity, input) => {
                        UiAction::SetParam(entity, input, ParamValue::Data(*handle))
                    }
                    Target::Filter(id, input) => {
                        UiAction::SetFilterParam(id, input, ParamValue::Data(*handle))
                    }
                    // The actor was waiting on this handle to exist, so make it
                    // now, bound. It goes through the same binding check as any
                    // other, and passes because the one thing it required is
                    // exactly what just arrived.
                    Target::NewActor {
                        object,
                        kind,
                        input,
                    } => {
                        let mut params = ParamMap::default();
                        params.insert(input.to_string(), ParamValue::Data(*handle));
                        UiAction::Create {
                            kind,
                            params,
                            making: Making::Actor(object),
                        }
                    }
                });
            }
            Ok(Err(refused)) => pending.error = Some(refused.to_string()),
            // Still in flight. The scene applies its commands on its own
            // schedule, so a reply is normally one tick out and sometimes more.
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => still_waiting.push(job),
            // The far end dropped it. Nothing to report and nothing to wait for.
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {}
        }
    }
    pending.waiting = still_waiting;
}
