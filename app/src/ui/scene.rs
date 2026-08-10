//! The Scene tab: the object tree, and what is drawn over it.
//!
//! Objects only. What draws them lives in [`super::actors`], because the tree
//! answers "where does this sit" and an actor's placement need not be where its
//! data came from.

use bevy_egui::egui;

use crate::viewport::OverlaySettings;

use super::gather::Gathered;
use super::{PendingActions, UiAction, UiState, human_bytes};

pub fn list(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    actions: &mut PendingActions,
    overlays: &mut OverlaySettings,
) {
    if scene.roots.is_empty() {
        ui.weak("Nothing loaded. Upload over gRPC.");
    }
    for root in &scene.roots {
        node(ui, *root, scene, state, actions);
    }

    ui.add_space(8.0);
    egui::CollapsingHeader::new("Overlays")
        .default_open(false)
        .show(ui, |ui| {
            ui.checkbox(&mut overlays.grid, "Ground grid");
            ui.checkbox(&mut overlays.world_axes, "World axes");
            ui.checkbox(&mut overlays.selection, "Selection outline");
            ui.checkbox(&mut overlays.all_bounds, "All bounds");
        });
}

/// One object and its children.
///
/// A leaf gets a plain selectable row rather than a collapsing header, so the
/// tree does not fill up with disclosure triangles that expand into nothing.
fn node(
    ui: &mut egui::Ui,
    entity: bevy::prelude::Entity,
    scene: &Gathered,
    state: &UiState,
    actions: &mut PendingActions,
) {
    let Some(row) = scene.rows.get(&entity) else {
        return;
    };
    let selected = state.selected == Some(entity);
    let label = if row.visible {
        format!("[{}] {}", row.id, row.name)
    } else {
        format!("[{}] {} (hidden)", row.id, row.name)
    };

    if row.children.is_empty() {
        if ui.selectable_label(selected, label).clicked() {
            actions.0.push(UiAction::Select(entity));
        }
        return;
    }

    egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        ui.make_persistent_id(entity),
        true,
    )
    .show_header(ui, |ui| {
        if ui.selectable_label(selected, label).clicked() {
            actions.0.push(UiAction::Select(entity));
        }
    })
    .body(|ui| {
        for child in &row.children {
            node(ui, *child, scene, state, actions);
        }
    });
}

pub fn details(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, actions: &mut PendingActions) {
    let Some(row) = state.selected.and_then(|entity| scene.rows.get(&entity)) else {
        ui.weak("Select an object.");
        return;
    };

    ui.heading(&row.name);
    ui.weak(format!("object {}", row.id));
    ui.separator();

    egui::Grid::new("object-details")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("kind");
            ui.monospace(row.kind.as_str());
            ui.end_row();
            ui.label("arrays");
            ui.monospace(format!("{}", row.arrays));
            ui.end_row();
            ui.label("size");
            ui.monospace(human_bytes(row.bytes));
            ui.end_row();
            ui.label("fields");
            ui.monospace(format!("{}", row.fields.len()));
            ui.end_row();
            ui.label("actors");
            ui.monospace(format!("{}", row.actors.len()));
            ui.end_row();
        });

    ui.separator();
    ui.horizontal(|ui| {
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
}
