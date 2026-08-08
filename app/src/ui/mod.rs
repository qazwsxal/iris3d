//! egui panels: menu bar, scene tree, and the array inventory.
//!
//! The UI reads the world and emits [`UiAction`]s rather than mutating
//! directly. Two reasons: egui closures would otherwise need several
//! conflicting mutable borrows at once, and keeping every mutation in one place
//! makes it obvious what the UI can actually do to the scene.

use bevy::asset::AssetId;
use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_egui::egui::{LayerId, Ui, UiBuilder};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui};

use crate::counter::UniqueID;
use crate::scene::data::{FieldKind, Fields};
use crate::scene::representation::ColorMap;
use crate::scene::{ColorBy, DataArray, DatasetKind, Representation, SceneCommand, SceneObject};
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

#[derive(Resource)]
pub struct UiState {
    pub show_scene: bool,
    pub show_arrays: bool,
    pub selected: Option<Entity>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            show_scene: true,
            show_arrays: true,
            selected: None,
        }
    }
}

/// Something the user asked for, applied after the UI has finished drawing.
enum UiAction {
    Select(Entity),
    ToggleVisibility(Entity),
    Delete(u64),
    Frame(Entity),
    FrameAll,
    /// Replace a representation entity's `Representation`.
    SetRepresentation(Entity, Representation),
    /// `None` paints flat; otherwise names a field in the parent's `Fields`.
    SetColourField(Entity, Option<String>),
    SetColourMap(Entity, ColorMap),
}

#[derive(Resource, Default)]
struct PendingActions(Vec<UiAction>);

/// A flattened view of one object, gathered before drawing so the UI closures
/// borrow nothing from the world.
struct Row {
    entity: Entity,
    id: u64,
    name: String,
    kind: DatasetKind,
    visible: bool,
    arrays: usize,
    bytes: u64,
    /// Field name and how many components it has, for the colour-by picker.
    fields: Vec<FieldRow>,
    representations: Vec<RepresentationRow>,
    children: Vec<Entity>,
}

struct FieldRow {
    name: String,
    kind: &'static str,
}

struct RepresentationRow {
    entity: Entity,
    representation: Representation,
    colour: ColorBy,
}

#[allow(clippy::too_many_arguments)]
fn draw_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<UiState>,
    mut actions: ResMut<PendingActions>,
    objects: Query<(
        Entity,
        &UniqueID,
        &SceneObject,
        &DatasetKind,
        &Visibility,
        Option<&Fields>,
        Option<&Children>,
        Option<&ChildOf>,
    )>,
    representations: Query<(&Representation, &ColorBy)>,
    arrays: Res<Assets<DataArray>>,
    mut captured: ResMut<PointerCaptured>,
    mut overlays: ResMut<crate::viewport::OverlaySettings>,
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
    let mut rows: HashMap<Entity, Row> = HashMap::new();
    let mut roots: Vec<Entity> = Vec::new();
    // Which object holds each array, for the inventory panel.
    let mut owners: HashMap<AssetId<DataArray>, (u64, String)> = HashMap::new();

    for (entity, id, object, kind, visibility, fields, children, parent) in &objects {
        let mut child_objects = Vec::new();
        let mut drawn = Vec::new();
        if let Some(children) = children {
            for child in children.iter() {
                if objects.contains(child) {
                    child_objects.push(child);
                } else if let Ok((representation, colour)) = representations.get(child) {
                    drawn.push(RepresentationRow {
                        entity: child,
                        representation: representation.clone(),
                        colour: colour.clone(),
                    });
                }
            }
        }

        for array in &object.arrays {
            owners.insert(array.handle.id(), (id.0, array.meta.name.clone()));
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
                representations: drawn,
                children: child_objects,
            },
        );
    }

    roots.sort_by_key(|entity| rows.get(entity).map(|row| row.id).unwrap_or_default());

    let total_bytes: u64 = rows.values().map(|row| row.bytes).sum();
    let object_count = rows.len();

    // Panel extents, so the 3D camera can be inset to whatever is left over.
    // Without this the scene renders across the whole window and hides behind
    // the panels.
    let mut edges = Vec4::ZERO;

    edges.y = egui::Panel::top("menu")
        .show(&mut root, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut state.show_scene, "Scene tree");
                ui.checkbox(&mut state.show_arrays, "Arrays");
                ui.separator();
                ui.checkbox(&mut overlays.grid, "Ground grid");
                ui.checkbox(&mut overlays.world_axes, "World axes");
                ui.checkbox(&mut overlays.selection, "Selection outline");
                ui.checkbox(&mut overlays.all_bounds, "All bounds");
            });
            ui.menu_button("Camera", |ui| {
                if ui.button("Frame all").clicked() {
                    actions.0.push(UiAction::FrameAll);
                    ui.close();
                }
            });
            ui.menu_button("Scene", |ui| {
                if ui.button("Delete all objects").clicked() {
                    for entity in &roots {
                        if let Some(row) = rows.get(entity) {
                            actions.0.push(UiAction::Delete(row.id));
                        }
                    }
                    ui.close();
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!(
                    "{object_count} objects · {} arrays · {}",
                    arrays.len(),
                    human_bytes(total_bytes)
                ));
            });
            });
        })
        .response
        .rect
        .height();

    if state.show_scene {
        edges.x = egui::Panel::left("scene")
            .resizable(true)
            .default_size(320.0)
            .show(&mut root, |ui| {
                ui.heading("Scene");
                ui.separator();
                if roots.is_empty() {
                    ui.weak("Nothing loaded. Upload over gRPC.");
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let roots = roots.clone();
                    for root in roots {
                        object_node(ui, root, &rows, &mut state, &mut actions);
                    }
                });
            })
            .response
            .rect
            .width();
    }

    if state.show_arrays {
        edges.z = egui::Panel::right("arrays")
            .resizable(true)
            .default_size(380.0)
            .show(&mut root, |ui| {
                ui.heading("Arrays");
                ui.weak(format!(
                    "{} in memory · {}",
                    arrays.len(),
                    human_bytes(total_bytes)
                ));
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("array-grid")
                        .num_columns(4)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("owner");
                            ui.strong("name");
                            ui.strong("type");
                            ui.strong("size");
                            ui.end_row();

                            let mut listing: Vec<_> = arrays.iter().collect();
                            listing.sort_by_key(|(id, _)| {
                                owners.get(id).map(|(handle, _)| *handle).unwrap_or(u64::MAX)
                            });

                            for (id, array) in listing {
                                let (owner, name) = owners
                                    .get(&id)
                                    .cloned()
                                    .unwrap_or((u64::MAX, "<unreferenced>".into()));
                                if owner == u64::MAX {
                                    ui.weak("—");
                                } else {
                                    ui.monospace(format!("{owner}"));
                                }
                                ui.label(name);
                                ui.monospace(format!(
                                    "{}{:?}",
                                    array.dtype,
                                    array.shape.iter().collect::<Vec<_>>()
                                ));
                                ui.monospace(human_bytes(array.data.len() as u64));
                                ui.end_row();
                            }
                        });
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
    let position = UVec2::new((edges.x * scale) as u32, (edges.y * scale) as u32);
    let full = UVec2::new(window.physical_width(), window.physical_height());
    let taken = position + UVec2::new((edges.z * scale) as u32, (edges.w * scale) as u32);
    camera.viewport = if full.cmpgt(taken).all() {
        Some(Viewport {
            physical_position: position,
            physical_size: full - taken,
            ..default()
        })
    } else {
        // Panels cover everything; a zero-sized viewport would panic.
        None
    };

    Ok(())
}

fn object_node(
    ui: &mut egui::Ui,
    entity: Entity,
    rows: &HashMap<Entity, Row>,
    state: &mut UiState,
    actions: &mut PendingActions,
) {
    let Some(row) = rows.get(&entity) else { return };
    let selected = state.selected == Some(entity);

    let header = egui::CollapsingHeader::new(
        egui::RichText::new(format!("[{}] {}", row.id, row.name)).strong(),
    )
    .id_salt(entity)
    .default_open(true);

    header.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(row.kind.as_str()).weak());
            ui.separator();
            ui.label(format!("{} arrays · {}", row.arrays, human_bytes(row.bytes)));
        });

        for representation in &row.representations {
            representation_controls(ui, row, representation, actions);
        }
        if row.representations.is_empty() && row.kind != DatasetKind::Empty {
            ui.label(egui::RichText::new("not drawn").weak());
        }

        ui.horizontal(|ui| {
            if ui
                .selectable_label(selected, if selected { "selected" } else { "select" })
                .clicked()
            {
                actions.0.push(UiAction::Select(row.entity));
            }
            if ui
                .button(if row.visible { "hide" } else { "show" })
                .clicked()
            {
                actions.0.push(UiAction::ToggleVisibility(row.entity));
            }
            if ui.button("frame").clicked() {
                actions.0.push(UiAction::Frame(row.entity));
            }
            if ui.button("delete").clicked() {
                actions.0.push(UiAction::Delete(row.id));
            }
        });

        for child in &row.children {
            object_node(ui, *child, rows, state, actions);
        }
    });
}

/// Representation picker, its parameters, and the colour-by controls.
fn representation_controls(
    ui: &mut egui::Ui,
    row: &Row,
    current: &RepresentationRow,
    actions: &mut PendingActions,
) {
    ui.group(|ui| {
        let options = Representation::available_for(row.kind);

        ui.horizontal(|ui| {
            ui.label("drawn as");
            egui::ComboBox::from_id_salt((current.entity, "rep"))
                .selected_text(current.representation.as_str())
                .show_ui(ui, |ui| {
                    for option in &options {
                        let picked = option.as_str() == current.representation.as_str();
                        if ui.selectable_label(picked, option.as_str()).clicked() && !picked {
                            actions
                                .0
                                .push(UiAction::SetRepresentation(current.entity, option.clone()));
                        }
                    }
                    // Shown but disabled: these variants exist in the model and
                    // have no backend, so offering them as choices would be a
                    // lie.
                    for pending in Representation::unimplemented_for(row.kind) {
                        ui.add_enabled(false, egui::Label::new(format!("{pending} (no backend)")));
                    }
                });
        });

        // Parameters for the current representation.
        match &current.representation {
            Representation::Points { size } => {
                let mut value = *size;
                if ui
                    .add(egui::Slider::new(&mut value, 0.001..=1.0).logarithmic(true).text("size"))
                    .changed()
                {
                    actions.0.push(UiAction::SetRepresentation(
                        current.entity,
                        Representation::Points { size: value },
                    ));
                }
            }
            Representation::BallAndStick {
                atom_scale,
                bond_radius,
            } => {
                let (mut atom, mut bond) = (*atom_scale, *bond_radius);
                let a = ui.add(egui::Slider::new(&mut atom, 0.05..=1.0).text("atom scale"));
                let b = ui.add(egui::Slider::new(&mut bond, 0.01..=0.5).text("bond radius"));
                if a.changed() || b.changed() {
                    actions.0.push(UiAction::SetRepresentation(
                        current.entity,
                        Representation::BallAndStick {
                            atom_scale: atom,
                            bond_radius: bond,
                        },
                    ));
                }
            }
            _ => {}
        }

        // Colour by.
        ui.horizontal(|ui| {
            ui.label("colour by");
            let selected = current.colour.field.clone().unwrap_or_else(|| "flat".into());
            egui::ComboBox::from_id_salt((current.entity, "colour"))
                .selected_text(selected.clone())
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(current.colour.field.is_none(), "flat")
                        .clicked()
                    {
                        actions
                            .0
                            .push(UiAction::SetColourField(current.entity, None));
                    }
                    for field in &row.fields {
                        let picked = current.colour.field.as_deref() == Some(field.name.as_str());
                        // Vector and tensor fields are reduced to magnitude, so
                        // say so rather than implying a direct mapping.
                        let label = if field.kind == "scalar" {
                            field.name.clone()
                        } else {
                            format!("{} ({} magnitude)", field.name, field.kind)
                        };
                        if ui.selectable_label(picked, label).clicked() && !picked {
                            actions.0.push(UiAction::SetColourField(
                                current.entity,
                                Some(field.name.clone()),
                            ));
                        }
                    }
                });
        });

        if current.colour.field.is_some() {
            ui.horizontal(|ui| {
                ui.label("map");
                for map in [ColorMap::Viridis, ColorMap::CoolWarm, ColorMap::Grayscale] {
                    let name = match map {
                        ColorMap::Viridis => "viridis",
                        ColorMap::CoolWarm => "cool-warm",
                        ColorMap::Grayscale => "grey",
                        ColorMap::ByElement => "element",
                    };
                    if ui
                        .selectable_label(current.colour.map == map, name)
                        .clicked()
                    {
                        actions.0.push(UiAction::SetColourMap(current.entity, map));
                    }
                }
            });
        }

        if row.fields.is_empty() {
            ui.label(egui::RichText::new("no fields").weak());
        }
    });
}

/// Applies what the UI asked for.
///
/// Deletion goes through [`SceneCommand`] rather than despawning directly, so
/// the UI takes exactly the same path a scripted client does — including the
/// non-recursive default that detaches children rather than destroying them.
fn apply_actions(
    mut actions: ResMut<PendingActions>,
    mut state: ResMut<UiState>,
    mut frame: ResMut<FrameRequest>,
    mut visibility: Query<&mut Visibility>,
    mut representations: Query<&mut Representation>,
    mut colours: Query<&mut ColorBy>,
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
            }
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
                let _ = bridge.sender().send(SceneCommand::DeleteObject {
                    id,
                    recursive: false,
                    reply,
                });
            }
            UiAction::Frame(entity) => frame.0 = Some(FrameTarget::Subtree(entity)),
            UiAction::FrameAll => frame.0 = Some(FrameTarget::All),

            // Each of these mutates a component the draw backends watch, so
            // `mark_dirty` picks them up and the geometry rebuilds. Nothing
            // here touches meshes directly.
            UiAction::SetRepresentation(entity, representation) => {
                if let Ok(mut current) = representations.get_mut(entity) {
                    *current = representation;
                }
            }
            UiAction::SetColourField(entity, field) => {
                if let Ok(mut colour) = colours.get_mut(entity) {
                    colour.field = field;
                }
            }
            UiAction::SetColourMap(entity, map) => {
                if let Ok(mut colour) = colours.get_mut(entity) {
                    colour.map = map;
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
