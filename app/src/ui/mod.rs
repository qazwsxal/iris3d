//! The egui interface: a menu bar, and one panel down the right-hand side.
//!
//! The panel carries three tabs — Data, Actors, Scene — and each is split into
//! a list on top and a details section below it. Everything used to be spread
//! across two side panels with the actor controls buried inside the tree, which
//! meant tuning a slider squeezed the 3D view from both sides at once. One
//! panel leaves the viewport the whole left of the window.
//!
//! The UI reads the world and emits [`UiAction`]s rather than mutating
//! directly. Two reasons: egui closures would otherwise need several
//! conflicting mutable borrows at once, and keeping every mutation in one place
//! makes it obvious what the UI can actually do to the scene. Splitting the
//! tabs into modules of their own tightens that constraint rather than
//! loosening it — a tab gets `&Gathered` and a queue to push onto, nothing
//! more.

mod actors;
mod data;
mod filters;
mod gather;
mod nodes;
mod params;
mod scene;

use bevy::asset::AssetId;
use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_egui::egui::{LayerId, Ui, UiBuilder};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui};

use crate::filter::FilterSummary;

use crate::scene::registry::{ActorKindId, ActorParams, ActorRegistry, ParamMap, ParamValue};
use crate::scene::{DataArray, SceneCommand, SceneError};
use crate::viewport::manipulate::GizmoMode;
use crate::viewport::{FrameRequest, FrameTarget, PointerCaptured};


pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            // Without this, bevy_egui attaches a context to the first camera
            // it finds — the 3D one — while `spawn_egui_camera` adds a second.
            // Two `PrimaryEguiContext` entities makes bevy_egui's own
            // `Single` query panic. Which one wins is a Startup ordering race,
            // so it can appear to work until something unrelated reorders.
            .insert_resource(bevy_egui::EguiGlobalSettings {
                auto_create_primary_context: false,
                ..default()
            })
            .init_resource::<UiState>()
            .init_resource::<PendingActions>()
            .init_resource::<Pending>()
            .init_resource::<nodes::NodeGraph>()
            .add_systems(Startup, spawn_egui_camera)
            .add_systems(EguiPrimaryContextPass, draw_ui)
            // `collect_replies` before `apply_actions`, so a reply that lands
            // this tick queues its binding and that binding is applied in the
            // same tick rather than the next one.
            // `take_picks` first: a click in the 3D view becomes the same
            // `UiAction` a tree click emits, and is applied in the tick it
            // happened rather than the one after.
            .add_systems(
                Update,
                (take_picks, collect_replies, apply_actions).chain(),
            );
    }
}

/// Gives egui a camera of its own.
///
/// bevy_egui derives its drawing rect from the camera holding the context. If
/// that is the 3D camera, insetting the 3D viewport to make room for the panels
/// also shrinks egui — which shrinks the viewport again the next frame, and the
/// UI collapses inwards. A separate 2D camera decouples the two: egui always
/// covers the whole window, and the 3D viewport is free to occupy whatever the
/// panels leave.
///
/// The render layer matters: gizmos are drawn to *every* camera whose layers
/// overlap `GizmoConfig::render_layers` (layer 0 by default). Left on layer 0,
/// this camera draws its own flat copy of the grid, axes and bounding boxes —
/// a second set that does not turn with the 3D view. Moving it off layer 0
/// keeps the overlays on the 3D camera alone.
fn spawn_egui_camera(mut commands: Commands) {
    commands.spawn((
        Camera2d,
        Camera {
            // Draw over the 3D view without wiping it.
            order: 1,
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(EGUI_LAYER),
        PrimaryEguiContext,
    ));
}

/// A layer of its own for the egui camera, so it renders no world content.
const EGUI_LAYER: usize = 1;

/// Which of the two views the window is showing.
///
/// Two whole views rather than a third panel: the graph wants the window, and
/// the scene it describes is the same one the panel lists, so there is nothing
/// to see side by side.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// The 3D scene, with the tabbed panel down the right.
    Scene,
    /// The node canvas, filling the window.
    Nodes,
}

/// Which tab of the panel is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Data,
    Filters,
    Actors,
    Scene,
}

/// Something being assembled, before it exists.
///
/// Neither an actor nor a filter can be created empty and filled in afterwards:
/// both check that every required input is bound *before* anything is spawned,
/// so a `geometry` filter with no positions and a `surface` with no geometry are
/// refused alike. That is right — either one has nothing to do — but it means
/// the interface has to hold a half-built thing somewhere, and this is where.
///
/// The Actors tab used to duck this by spawning through `spawn_actor` directly,
/// which skips the check the command path runs. It could therefore make actors
/// a script could not: `add_actor("surface")` over gRPC is refused outright,
/// while the button beside it produced one that silently drew nothing. Same
/// draft for both closes that gap.
pub struct Draft {
    pub kind: &'static str,
    pub params: ParamMap,
    pub making: Making,
}

/// Which half of the split a draft is building.
#[derive(Clone, Copy)]
pub enum Making {
    /// Where to plug the finished filter in, once its handles are known.
    Filter(Option<(&'static str, Target)>),
    /// The object to draw it under.
    Actor(Entity),
}

/// Something already in the scene that a newly created filter should feed.
///
/// The two halves of the split need asking differently — an actor's parameters
/// are written straight onto its component, a filter's go back down the same
/// command channel a script uses — so which it is has to be carried along.
#[derive(Clone, Copy)]
pub enum Target {
    /// An actor entity and the input to bind.
    Actor(Entity, &'static str),
    /// A filter's handle and the input to bind.
    Filter(u64, &'static str),
    /// An actor that does not exist yet, to be created once the handle does.
    ///
    /// What "assemble…" means when it is clicked from an actor *draft*: the
    /// actor cannot be made before the geometry it requires, so the draft is
    /// carried along in here and spent when the filter lands. Without it the
    /// offer would have to nest one draft inside another.
    NewActor {
        object: Entity,
        kind: &'static str,
        input: &'static str,
    },
}

#[derive(Resource)]
pub struct UiState {
    pub view: View,
    pub tab: Tab,
    pub show_panel: bool,
    /// The selected object.
    ///
    /// Not tab-local, unlike the two below: `viewport::overlays` reads it to
    /// draw the selection outline, the Scene tree highlights it, and the Actors
    /// tab tints its group. Selecting an actor moves this to that actor's
    /// source so all three agree.
    pub selected: Option<Entity>,
    pub selected_actor: Option<Entity>,
    pub selected_array: Option<AssetId<DataArray>>,
    /// The selected filter, by handle. See [`Gathered::filter`].
    pub selected_filter: Option<u64>,
    /// The actor or filter being built, if a create form is open. One at a
    /// time, which is what keeps an offer from nesting a draft inside a draft.
    pub draft: Option<Draft>,
    /// What the transform handles do to the selected object.
    ///
    /// Here rather than in a resource of its own, so the viewport reads it from
    /// the same place it already reads the selection — and so `draw_ui` needs no
    /// seventeenth system parameter. See `viewport::manipulate`.
    pub gizmo: crate::viewport::manipulate::GizmoMode,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            view: View::Scene,
            tab: Tab::Scene,
            show_panel: true,
            selected: None,
            selected_actor: None,
            selected_array: None,
            selected_filter: None,
            draft: None,
            gizmo: crate::viewport::manipulate::GizmoMode::default(),
        }
    }
}

/// Something the user asked for, applied after the UI has finished drawing.
enum UiAction {
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
struct PendingActions(Vec<UiAction>);

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
struct Pending {
    waiting: Vec<Job>,
    /// Shown at the foot of the panel until the next thing succeeds.
    error: Option<String>,
}

struct Job {
    reply: tokio::sync::oneshot::Receiver<Result<FilterSummary, SceneError>>,
    /// Which output to bind, and where.
    then: Option<(&'static str, Target)>,
    /// Show the filter this reply is about once it lands.
    ///
    /// True for a creation and false for an edit. Without the distinction a
    /// slider drag — which is one `SetFilter` per frame — would drag the
    /// selection back to the filter being edited every frame, and take it off
    /// whatever the user clicked in the middle of the drag.
    select: bool,
}

#[allow(clippy::too_many_arguments)]
fn draw_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<UiState>,
    mut actions: ResMut<PendingActions>,
    read: gather::SceneRead,
    arrays: Res<Assets<DataArray>>,
    mut captured: ResMut<PointerCaptured>,
    mut overlays: ResMut<crate::viewport::OverlaySettings>,
    pending: Res<Pending>,
    mut graph: ResMut<nodes::NodeGraph>,
    // Filtered on Camera3d, not `Without<EguiContext>` as bevy_egui's own
    // example does: with a single camera, bevy_egui puts the egui context on
    // that same entity, so excluding it matches nothing. A `Single` param that
    // matches nothing makes Bevy skip the system silently — no panels, no
    // error, nothing in the log.
    mut cameras: Query<&mut Camera, With<Camera3d>>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    captured.0 = ctx.egui_wants_pointer_input();

    // egui 0.35 panels attach to a `Ui` rather than a `Context`, so the root
    // Ui covering the whole viewport has to be built explicitly.
    let mut root = Ui::new(
        ctx.clone(),
        "iris3d-root".into(),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    // Gather first, draw second.
    let world = gather::gather(&read);

    // How much the panels took, so the 3D camera can be inset to what is left.
    // Without this the scene renders across the whole window and hides behind
    // them.
    let top = egui::Panel::top("menu")
        .show(&mut root, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut state.show_panel, "Panel");
                });
                ui.menu_button("Camera", |ui| {
                    if ui.button("Frame all").clicked() {
                        actions.0.push(UiAction::FrameAll);
                        ui.close();
                    }
                });
                ui.menu_button("Scene", |ui| {
                    if ui.button("Delete all objects").clicked() {
                        for entity in &world.roots {
                            if let Some(row) = world.rows.get(entity) {
                                actions.0.push(UiAction::Delete(row.id));
                            }
                        }
                        ui.close();
                    }
                });
                // What the drag handles do, beside the view toggle rather than
                // in a menu: it is a mode, and a mode that is two clicks away
                // reads as a setting. Only shown in the Scene view, since the
                // node canvas has no handles to switch between.
                if state.view == View::Scene {
                    ui.separator();
                    for (mode, label) in [
                        (GizmoMode::Translate, "Move"),
                        (GizmoMode::Rotate, "Turn"),
                        (GizmoMode::Scale, "Size"),
                    ] {
                        if ui
                            .selectable_label(state.gizmo == mode, label)
                            .clicked()
                        {
                            state.gizmo = mode;
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Right-to-left, so this is the rightmost thing in the bar.
                    let (label, to) = match state.view {
                        View::Scene => ("Nodes", View::Nodes),
                        View::Nodes => ("Scene", View::Scene),
                    };
                    if ui.button(label).clicked() {
                        state.view = to;
                    }
                    ui.separator();
                    // The meshes are counted apart from the arrays because
                    // their vertices are on the GPU rather than in
                    // `Assets<DataArray>` — and because this is where sharing
                    // shows: drawing one ribbon as a lit surface *and* as an
                    // absorbing medium should add an actor and no vertices.
                    let mut summary = format!(
                        "{} objects · {} arrays · {}",
                        world.rows.len(),
                        arrays.len(),
                        human_bytes(world.total_bytes)
                    );
                    if world.meshes > 0 {
                        summary.push_str(&format!(
                            " · {} mesh{}, {} verts",
                            world.meshes,
                            if world.meshes == 1 { "" } else { "es" },
                            world.vertices
                        ));
                    }
                    ui.label(summary);
                });
            });
        })
        .response
        .rect
        .height();

    // The node view takes the window. Drawn before the panel and returning
    // early, so `right` stays zero and the 3D camera is given nothing below —
    // there is no scene on screen to inset it into.
    if state.view == View::Nodes {
        nodes::show(&mut root, &mut graph, &world, &mut actions, &state);
        // Nothing of the 3D scene is on screen, so the camera is given no
        // viewport at all rather than a sliver behind the canvas.
        if let Ok(mut camera) = cameras.single_mut() {
            camera.viewport = None;
        }
        return Ok(());
    }

    let mut right = 0.0;
    if state.show_panel {
        right = egui::Panel::right("panel")
            .resizable(true)
            .default_size(380.0)
            .show(&mut root, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Data, "Data"),
                        (Tab::Filters, "Filters"),
                        (Tab::Actors, "Actors"),
                        (Tab::Scene, "Scene"),
                    ] {
                        if ui.selectable_label(state.tab == tab, label).clicked() {
                            state.tab = tab;
                        }
                    }
                });
                ui.separator();

                // Details before the list. An egui panel claims its space out
                // of what is currently available and leaves the rest to
                // whatever is added afterwards, so a list added first would
                // take the lot and leave the details nothing to sit in.
                let tab = state.tab;
                egui::Panel::bottom("details")
                    .resizable(true)
                    .default_size(280.0)
                    .min_size(60.0)
                    .show(ui, |ui| {
                        // A resizable panel is only as tall as content that
                        // fills it, and a `ScrollArea` shrinks to fit by
                        // default — which collapsed this to a single line
                        // whenever nothing was selected. Refusing to shrink is
                        // what makes the split hold its height.
                        egui::ScrollArea::vertical()
                            .id_salt("details")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                match tab {
                                    Tab::Data => data::details(ui, &world, &state, &arrays),
                                    Tab::Filters => filters::details(
                                        ui,
                                        &world,
                                        &state,
                                        &read.filter_registry,
                                        &mut actions,
                                    ),
                                    Tab::Actors => {
                                        actors::details(
                                            ui,
                                            &world,
                                            &state,
                                            &read.registry,
                                            &mut actions,
                                        )
                                    }
                                    Tab::Scene => scene::details(ui, &world, &state, &mut actions),
                                }
                                // Whatever the backend last refused. Filters are
                                // the only thing here that can be refused for a
                                // reason worth reading — a binding that does not
                                // fit, a cycle — and the reply that carries it is
                                // otherwise dropped on the floor.
                                if let Some(message) = &pending.error {
                                    ui.separator();
                                    ui.colored_label(ui.visuals().error_fg_color, message);
                                }
                            });
                    });

                egui::ScrollArea::vertical()
                    .id_salt("list")
                    .show(ui, |ui| match tab {
                        Tab::Data => data::list(ui, &world, &state, &mut actions, &arrays),
                        Tab::Filters => filters::list(ui, &world, &state, &mut actions),
                        Tab::Actors => actors::list(ui, &world, &state, &mut actions),
                        Tab::Scene => scene::list(
                            ui,
                            &world,
                            &state,
                            &mut actions,
                            &mut overlays,
                        ),
                    });
            })
            .response
            .rect
            .width();
    }

    // Inset the 3D camera to the area the panels left free.
    let (Ok(window), Ok(mut camera)) = (windows.single(), cameras.single_mut()) else {
        return Ok(());
    };
    let scale = window.scale_factor();
    let position = UVec2::new(0, (top * scale) as u32);
    let full = UVec2::new(window.physical_width(), window.physical_height());
    let taken = position + UVec2::new((right * scale) as u32, 0);
    camera.viewport = if full.cmpgt(taken).all() {
        Some(Viewport {
            physical_position: position,
            physical_size: full - taken,
            ..default()
        })
    } else {
        // The panel covers everything; a zero-sized viewport would panic.
        None
    };

    Ok(())
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
fn take_picks(
    mut picks: MessageReader<crate::viewport::pick::Picked>,
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

fn apply_actions(
    mut commands: Commands,
    mut actions: ResMut<PendingActions>,
    mut state: ResMut<UiState>,
    mut frame: ResMut<FrameRequest>,
    registry: Res<ActorRegistry>,
    filter_registry: Res<crate::filter::FilterRegistry>,
    mut visibility: Query<&mut Visibility>,
    mut params: Query<(&ActorKindId, &mut ActorParams)>,
    actor_entities: Query<(), With<ActorKindId>>,
    // The handle as well as the membership: an actor is asked for by object
    // handle, because that is the name the command speaks in.
    scene_objects: Query<&crate::counter::UniqueID, With<crate::scene::SceneObject>>,
    bridge: Res<crate::grpc::GrpcBridge>,
    mut pending: ResMut<Pending>,
) {
    for action in actions.0.drain(..) {
        match action {
            UiAction::Select(entity) => {
                state.selected = if state.selected == Some(entity) {
                    None
                } else {
                    Some(entity)
                };
                // The Actors tab would otherwise keep showing the controls for
                // an actor of whatever was selected before, tinted under a
                // group that is no longer highlighted.
                state.selected_actor = None;
            }
            UiAction::SelectActor(entity, under) => {
                state.selected_actor = Some(entity);
                // Select the object it was clicked under too: the outline, the
                // tree highlight and the tint in the actor list all key off the
                // object selection, and three of them disagreeing is worse than
                // the outline moving. An actor drawn nowhere clears it rather
                // than leaving it on whatever was picked before.
                state.selected = under.filter(|parent| scene_objects.contains(*parent));
            }
            UiAction::SelectArray(id) => state.selected_array = Some(id),
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
                let _ = bridge
                    .sender()
                    .send(SceneCommand::DeleteObject { id, reply });
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
                    if state.selected_actor == Some(entity) {
                        state.selected_actor = None;
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
                let _ = bridge.sender().send(SceneCommand::SetActor {
                    id,
                    params: ParamMap::default(),
                    visible: None,
                    parents: Some(parents),
                    reply,
                });
            }
            UiAction::SetObjectParent(id, parent) => {
                let (reply, _) = tokio::sync::oneshot::channel();
                let _ = bridge.sender().send(SceneCommand::SetParent {
                    id,
                    parent,
                    // The object stays where it looks like it is. Re-parenting
                    // by dragging a wire says nothing about wanting to move it,
                    // and having it jump would be a surprise.
                    keep_world_transform: true,
                    reply,
                });
            }

            UiAction::SelectFilter(id) => state.selected_filter = Some(id),

            // Filters go down the command channel rather than being written
            // here. Not for consistency's sake: `set` merges the change over the
            // current map, re-checks every binding and refuses a cycle, and
            // duplicating that in the interface would be a second, worse copy of
            // the rules.
            UiAction::SetFilterParam(id, input, value) => {
                let mut params = ParamMap::default();
                params.insert(input.to_string(), value);
                let (reply, receiver) = tokio::sync::oneshot::channel();
                if bridge
                    .sender()
                    .send(SceneCommand::SetFilter { id, params, reply })
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
                let _ = bridge
                    .sender()
                    .send(SceneCommand::RemoveFilter { id, reply });
                state.selected_filter = None;
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
                        if bridge
                            .sender()
                            .send(SceneCommand::AddFilter {
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
                        let _ = bridge.sender().send(SceneCommand::AddActor {
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
fn collect_replies(mut pending: ResMut<Pending>, mut actions: ResMut<PendingActions>) {
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
                let Some((_, handle)) = summary
                    .outputs
                    .iter()
                    .find(|(id, _)| id == output)
                else {
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

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
