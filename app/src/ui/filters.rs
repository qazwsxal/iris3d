//! The Filters tab: everything deriving data, and the controls for one of them.
//!
//! A filter reads arrays and writes arrays or one mesh, and draws nothing. That
//! is the half of the scene this interface had no way to see or touch at all —
//! and since geometry became a filter's job, the half without which an uploaded
//! mesh cannot be put on screen.
//!
//! The list is flat and in handle order, unlike the Actors tab's grouping by
//! object. A filter belongs to no object; it is reached through the data it
//! writes. What replaces the grouping is saying, on every input and every
//! output, which filter is on the other end — a chain reads as a chain that way,
//! without a graph view to draw it.

use bevy_egui::egui;

use crate::filter::FilterRegistry;

use super::gather::{FilterRow, Gathered};
use super::params;
use super::{PendingActions, Target, UiAction, UiState};

pub fn list(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, actions: &mut PendingActions) {
    if scene.filters.is_empty() {
        ui.weak("No filters. Nothing is deriving anything.");
        return;
    }
    for filter in &scene.filters {
        ui.horizontal(|ui| {
            let picked = state.selected_filter == Some(filter.id);
            if ui
                .selectable_label(picked, format!("[{}] {}", filter.id, filter.label))
                .clicked()
            {
                actions.0.push(UiAction::SelectFilter(filter.id));
            }
            // Both states, not one. A chain costs a frame per link, so a filter
            // three deep is briefly stale before it is busy, and that is it
            // working rather than it stuck.
            if filter.busy {
                ui.label(egui::RichText::new("running").weak());
            } else if filter.stale {
                ui.label(egui::RichText::new("stale").weak());
            }
        });
    }
}

pub fn details(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    registry: &FilterRegistry,
    actions: &mut PendingActions,
) {
    match state.selected_filter.and_then(|id| scene.filter(id)) {
        Some(filter) => controls(ui, scene, filter, actions),
        None => {
            ui.weak("Select a filter.");
        }
    }
    ui.separator();
    draft(ui, scene, state, registry, actions);
}

/// One filter: what it reads, what it writes, and who wants it.
fn controls(
    ui: &mut egui::Ui,
    scene: &Gathered,
    current: &FilterRow,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        ui.heading(current.label);
        ui.weak(format!("[{}]", current.id));
        if ui.small_button("remove").clicked() {
            actions.0.push(UiAction::RemoveFilter(current.id));
        }
    });
    ui.separator();

    // `geometry`'s colour input is the one place a second filter is the obvious
    // next step: colour on a shared vertex buffer has to be mapped before the
    // buffer is assembled, so "colour by this field" is a `colormap` upstream of
    // here rather than a setting on anything downstream.
    let edits = params::controls(
        ui,
        scene,
        current.specs,
        &current.params,
        current.entity,
        |spec| (current.kind == "geometry" && spec.id == "colour").then_some("colour by…"),
    );
    for (input, value) in edits.set {
        actions
            .0
            .push(UiAction::SetFilterParam(current.id, input, value));
    }
    if let Some(input) = edits.offered {
        actions.0.push(UiAction::OfferFilter {
            kind: "colormap",
            then: Some(("colour", Target::Filter(current.id, input))),
        });
    }

    ui.separator();
    ui.label(egui::RichText::new("writes").strong());
    for (spec, handle) in &current.outputs {
        ui.horizontal(|ui| {
            ui.label(spec.label);
            let described = scene
                .bindable
                .iter()
                .find(|(id, _)| id == handle)
                .map(|(_, meta)| meta.describe())
                .unwrap_or_else(|| "not held".into());
            ui.monospace(format!("d{handle} · {described}"));
        });
        // An output nothing reads is a dead branch. Saying so is the difference
        // between a chain that is half-built and one that merely looks it.
        match scene.consumers.get(handle) {
            Some(readers) => {
                for reader in readers {
                    ui.weak(format!(
                        "        → [{}] {} · {}",
                        reader.id, reader.label, reader.input
                    ));
                }
            }
            None => {
                ui.weak("        → nothing reads this");
            }
        }
    }
}

/// The create form.
///
/// A staged form rather than a button, because a filter cannot be made empty:
/// `filter::add` checks that every required input is bound before it spawns
/// anything. So the bindings are chosen here, against a draft that exists only
/// in the interface, and the filter is asked for once it would be accepted.
fn draft(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    registry: &FilterRegistry,
    actions: &mut PendingActions,
) {
    let Some(draft) = &state.draft else {
        ui.horizontal(|ui| {
            ui.label("add a filter");
            egui::ComboBox::from_id_salt("add-filter")
                .selected_text("choose a kind")
                .show_ui(ui, |ui| {
                    for kind in registry.iter() {
                        if ui.selectable_label(false, kind.label).clicked() {
                            actions.0.push(UiAction::OfferFilter {
                                kind: kind.id,
                                then: None,
                            });
                        }
                    }
                });
        });
        return;
    };

    let Some(kind) = registry.get(draft.kind) else {
        return;
    };
    // No offers inside a filter draft: the only one that makes sense is
    // "assemble geometry", and a filter never takes geometry as an input.
    params::draft_form(ui, scene, actions, draft, kind.label, kind.params, |_| None);
}
