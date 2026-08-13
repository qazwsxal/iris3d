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
mod gather;
mod scene;

use bevy::asset::AssetId;
use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_egui::egui::{LayerId, Ui, UiBuilder};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui};

use crate::scene::actor::ColorMap;
use crate::scene::link::spawn_actor;
use crate::scene::registry::{ActorKindId, ActorParams, ActorRegistry, ParamValue};
use crate::scene::{ColorBy, DataArray, Placement, SceneCommand, Subset};
use crate::viewport::{FrameRequest, FrameTarget, PointerCaptured};

use gather::{ActorData, ObjectData};

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
            .add_systems(Startup, spawn_egui_camera)
            .add_systems(EguiPrimaryContextPass, draw_ui)
            .add_systems(Update, apply_actions);
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

/// Which tab of the panel is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Data,
    Actors,
    Scene,
}

#[derive(Resource)]
pub struct UiState {
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
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            tab: Tab::Scene,
            show_panel: true,
            selected: None,
            selected_actor: None,
            selected_array: None,
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
    /// Draw an object an additional way. The object keeps the actors it
    /// already has — this adds, it does not replace.
    AddActor(Entity, &'static str),
    RemoveActor(Entity),
    /// Change one parameter of an actor, leaving the rest alone.
    SetParam(Entity, &'static str, ParamValue),
    SetColourMap(Entity, ColorMap),
    /// The colour used when no array is bound to the kind's colour input.
    ///
    /// What that *means* is the kind's business, and it differs more than the
    /// name suggests: for a mesh or a point cloud it is simply what the thing
    /// is painted, but the moment backend reads it as the medium's
    /// transmission — the fraction of each channel a volume lets through — and
    /// reads it whether or not a colour array is bound. So this is not a
    /// cosmetic control everywhere; on a volume it decides how much is
    /// absorbed.
    SetColourFlat(Entity, Color),
    /// Where the colour map starts and ends. `None` autoscales to the bound
    /// array's own extremes, which is the default and usually what is wanted;
    /// pinning it is what makes two actors comparable to each other.
    SetColourRange(Entity, Option<(f32, f32)>),
}

#[derive(Resource, Default)]
struct PendingActions(Vec<UiAction>);

#[allow(clippy::too_many_arguments)]
fn draw_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<UiState>,
    mut actions: ResMut<PendingActions>,
    objects: ObjectData,
    actor_data: ActorData,
    placements: Query<&Placement>,
    registry: Res<ActorRegistry>,
    arrays: Res<Assets<DataArray>>,
    store: Res<crate::scene::DataStore>,
    mut captured: ResMut<PointerCaptured>,
    mut overlays: ResMut<crate::viewport::OverlaySettings>,
    // Only the raytracing pathway inserts this, so the UI asks rather than
    // assumes. See `scene::list`.
    mut accumulation: Option<ResMut<crate::draw::Accumulation>>,
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
    let world = gather::gather(&objects, &actor_data, &placements, &registry, &store);

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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!(
                        "{} objects · {} arrays · {}",
                        world.rows.len(),
                        arrays.len(),
                        human_bytes(world.total_bytes)
                    ));
                });
            });
        })
        .response
        .rect
        .height();

    let mut right = 0.0;
    if state.show_panel {
        right = egui::Panel::right("panel")
            .resizable(true)
            .default_size(380.0)
            .show(&mut root, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Data, "Data"),
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
                            .show(ui, |ui| match tab {
                                Tab::Data => data::details(ui, &world, &state, &arrays),
                                Tab::Actors => actors::details(ui, &world, &state, &mut actions),
                                Tab::Scene => scene::details(ui, &world, &state, &mut actions),
                            });
                    });

                egui::ScrollArea::vertical()
                    .id_salt("list")
                    .show(ui, |ui| match tab {
                        Tab::Data => data::list(ui, &world, &state, &mut actions, &arrays),
                        Tab::Actors => actors::list(ui, &world, &state, &mut actions),
                        Tab::Scene => scene::list(
                            ui,
                            &world,
                            &state,
                            &mut actions,
                            &mut overlays,
                            accumulation.as_deref_mut(),
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
fn apply_actions(
    mut commands: Commands,
    mut actions: ResMut<PendingActions>,
    mut state: ResMut<UiState>,
    mut frame: ResMut<FrameRequest>,
    mut counter: ResMut<crate::counter::GlobalIDCounter>,
    registry: Res<ActorRegistry>,
    mut visibility: Query<&mut Visibility>,
    mut params: Query<(&ActorKindId, &mut ActorParams)>,
    mut colours: Query<&mut ColorBy>,
    actor_entities: Query<(), With<ActorKindId>>,
    scene_objects: Query<(), With<crate::scene::SceneObject>>,
    bridge: Res<crate::grpc::GrpcBridge>,
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
                let (_, actor) = spawn_actor(
                    &mut commands,
                    &mut counter,
                    // One object here. Drawing it somewhere else as well is a
                    // second parent, which a client asks for by handle — there
                    // is nothing to click on to mean "and also there".
                    vec![object],
                    // Selections are computed by a client, not clicked together
                    // here, so the tree only ever adds a whole-dataset one.
                    Subset::All,
                    (
                        ActorKindId(registered.id),
                        ActorParams(registered.defaults()),
                        ColorBy::default(),
                    ),
                );
                // Show what was just added, so its controls appear without a
                // second click into the list.
                state.selected_actor = Some(actor);
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
            UiAction::SetColourMap(entity, map) => {
                if let Ok(mut colour) = colours.get_mut(entity) {
                    colour.map = map;
                }
            }
            UiAction::SetColourFlat(entity, flat) => {
                if let Ok(mut colour) = colours.get_mut(entity) {
                    colour.flat = flat;
                }
            }
            UiAction::SetColourRange(entity, range) => {
                if let Ok(mut colour) = colours.get_mut(entity) {
                    colour.range = range;
                }
            }
        }
    }
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
