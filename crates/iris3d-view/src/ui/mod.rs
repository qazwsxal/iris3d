//! The egui interface: a menu bar, and one panel down the right-hand side.
//!
//! Two views, switched from the menu bar. The **scene** view puts the panel
//! beside the 3D viewport; the **nodes** view gives the graph canvas the whole
//! window. See [`View`].
//!
//! The panel carries four tabs — Data, Filters, Actors, Scene — and each is
//! split into a list on top and a details section below it. One panel rather
//! than two, so the viewport keeps the whole left of the window and tuning a
//! slider does not squeeze it from both sides.
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

use iris3d_filter::{FilterBus, FilterCommand, FilterSummary};

use crate::select::Selection;
use crate::viewport::manipulate::GizmoMode;
use crate::viewport::{FrameRequest, FrameTarget, PointerCaptured};
use iris3d_model::SceneError;
use iris3d_model::{ParamMap, ParamValue};
use iris3d_scene::registry::{ActorKindId, ActorParams, ActorRegistry};
use iris3d_scene::{DataArray, SceneCommand};

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
            .add_systems(Update, (take_picks, collect_replies, apply_actions).chain());
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
/// Both halves of the interface build through the same draft and the same
/// command path, so a button can never make an actor a script could not.
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
    /// The actor or filter being built, if a create form is open. One at a
    /// time, which is what keeps an offer from nesting a draft inside a draft.
    pub draft: Option<Draft>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            view: View::Scene,
            tab: Tab::Scene,
            show_panel: true,
            draft: None,
        }
    }
}

mod action;
mod panel;

use action::{Pending, apply_actions, collect_replies, take_picks};
pub use action::{PendingActions, UiAction};
use panel::draw_ui;

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
