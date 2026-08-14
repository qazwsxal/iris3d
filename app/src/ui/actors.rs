//! The Actors tab: everything being drawn, and the controls for one of them.
//!
//! The list is the whole scene rather than just the selected object's actors.
//! Two actors of one object differ only in their settings, and comparing them
//! against a third somewhere else is the reason several exist at all — hiding
//! the rest behind a selection would make that impossible to see. The selected
//! object's group is tinted instead.

use bevy_egui::egui;

use crate::scene::Subset;
use crate::scene::registry::{
    ParamKind, ParamValue, data as registry_data, flag, float, text, vector as registry_vector,
};

use super::gather::{ActorRow, Gathered, Row};
use super::{PendingActions, UiAction, UiState};

pub fn list(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, actions: &mut PendingActions) {
    let mut drew_anything = false;

    for object in &scene.ordered {
        let Some(row) = scene.rows.get(object) else {
            continue;
        };
        if row.actors.is_empty() {
            continue;
        }
        drew_anything = true;

        let highlighted = state.selected == Some(row.entity);
        let frame = if highlighted {
            // The selection colour rather than a fixed grey, so this follows
            // whichever theme egui is in.
            egui::Frame::new().fill(ui.visuals().selection.bg_fill.gamma_multiply(0.25))
        } else {
            egui::Frame::new()
        };

        frame.inner_margin(4.0).show(ui, |ui| {
            let heading = egui::RichText::new(format!("[{}] {}", row.id, row.name));
            ui.label(if highlighted {
                heading.strong()
            } else {
                heading.weak()
            });
            for actor in &row.actors {
                entry(ui, actor, Some(row), state, actions);
            }
        });
    }

    // Actors drawn under nothing, usually because the last object they were
    // under was deleted. Grouped on their own because they are in no object's
    // list — leaving them out would make them unreachable, with nothing on
    // screen to say they still exist.
    if !scene.detached.is_empty() {
        drew_anything = true;
        egui::Frame::new().inner_margin(4.0).show(ui, |ui| {
            ui.label(egui::RichText::new("Detached — not drawn").weak().italics());
            for actor in &scene.detached {
                entry(ui, actor, None, state, actions);
            }
        });
    }

    if !drew_anything {
        ui.weak("Nothing is drawn.");
    }
}

/// One clickable row in the list.
///
/// `under` is the object whose group this row is in, which is what the click
/// selects alongside the actor. One actor drawn under several objects has a row
/// in each of their groups, and they differ only in this.
fn entry(
    ui: &mut egui::Ui,
    actor: &ActorRow,
    under: Option<&Row>,
    state: &UiState,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        let picked = state.selected_actor == Some(actor.entity);
        if ui
            .selectable_label(picked, format!("[{}] {}", actor.id, actor.label))
            .clicked()
        {
            actions
                .0
                .push(UiAction::SelectActor(actor.entity, under.map(|r| r.entity)));
        }
        // Worth saying outright: two identical-looking rows over one object are
        // otherwise indistinguishable when what differs is which part of the
        // data each draws.
        if matches!(actor.subset, Subset::Selected { .. }) {
            ui.label(egui::RichText::new("subset").weak());
        }
    });
}

pub fn details(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, actions: &mut PendingActions) {
    match state.selected_actor.and_then(|entity| scene.actor(entity)) {
        Some((row, actor)) => {
            controls(ui, scene, row, actor, actions);
            ui.separator();
        }
        None => {
            ui.weak("Select an actor.");
            ui.separator();
        }
    }
    add(ui, scene, state, actions);
}

/// The "draw this another way" row.
///
/// Adding rather than replacing: an object may be drawn several ways at once,
/// and each way is its own entity with its own settings.
fn add(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, actions: &mut PendingActions) {
    let Some(row) = state.selected.and_then(|entity| scene.rows.get(&entity)) else {
        ui.weak("Select an object to draw.");
        return;
    };
    if row.available.is_empty() {
        ui.weak("This build has no registered actor kinds.");
        return;
    }
    ui.horizontal(|ui| {
        ui.label(format!("add to [{}] {}", row.id, row.name));
        egui::ComboBox::from_id_salt((row.entity, "add"))
            .selected_text("choose a kind")
            .show_ui(ui, |ui| {
                for kind in &row.available {
                    if ui.selectable_label(false, kind.label).clicked() {
                        actions.0.push(UiAction::AddActor(row.entity, kind.id));
                    }
                }
            });
    });
}

/// One actor: what it is and its parameters.
///
/// The controls are generated from the backend's own `ParamSpec` declarations,
/// so adding an actor kind — or a parameter to an existing one — needs no edit
/// here. A slider's range is the declared range, which is also the range values
/// are clamped to on the way in, so the UI cannot ask for something a client
/// could not.
fn controls(
    ui: &mut egui::Ui,
    scene: &Gathered,
    row: Option<&Row>,
    current: &ActorRow,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        ui.heading(current.label);
        match row {
            Some(row) => ui.weak(format!("of [{}] {}", row.id, row.name)),
            // Nowhere to be drawn, so nothing is on screen. Worth saying
            // outright — otherwise these controls look broken.
            None => ui.weak("under no object — not drawn"),
        };
    });
    // One actor, several places. Every control below changes all of them at
    // once, which is the reason to draw it this way rather than as two actors.
    if current.places > 1 {
        ui.weak(format!("drawn under {} objects", current.places));
    }
    ui.horizontal(|ui| {
        if matches!(current.subset, Subset::Selected { .. }) {
            ui.label(egui::RichText::new("subset").weak());
        }
        if ui.small_button("remove").clicked() {
            actions.0.push(UiAction::RemoveActor(current.entity));
        }
    });
    ui.separator();

    for spec in current.specs {
        match spec.kind {
            ParamKind::Float {
                default,
                min,
                max,
                logarithmic,
            } => {
                let mut value = float(&current.params, spec.id, default);
                if ui
                    .add(
                        egui::Slider::new(&mut value, min..=max)
                            .logarithmic(logarithmic)
                            .text(spec.label),
                    )
                    .changed()
                {
                    actions.0.push(UiAction::SetParam(
                        current.entity,
                        spec.id,
                        ParamValue::Float(value),
                    ));
                }
            }
            ParamKind::Vector {
                components,
                min,
                max,
                integral,
                ..
            } => {
                // One drag value per component. Labelled x/y/z up to three and
                // then by index, so a two-component parameter reads correctly
                // without this needing to know what it is for.
                let mut values = registry_vector(&current.params, spec.id, components);
                let mut edited = false;
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    for (axis, value) in values.iter_mut().enumerate() {
                        let name = ["x", "y", "z", "w"]
                            .get(axis)
                            .copied()
                            .map(str::to_string)
                            .unwrap_or_else(|| axis.to_string());
                        let speed = if integral { 1.0 } else { 0.01 };
                        if ui
                            .add(
                                egui::DragValue::new(value)
                                    .speed(speed)
                                    .range(min..=max)
                                    .prefix(format!("{name} ")),
                            )
                            .changed()
                        {
                            edited = true;
                        }
                    }
                });
                if edited {
                    actions.0.push(UiAction::SetParam(
                        current.entity,
                        spec.id,
                        ParamValue::Vector(values),
                    ));
                }
            }
            ParamKind::Array { required, .. } => {
                // Generated from the same declaration as every other control,
                // which is the point of arrays being parameters: the picker only
                // offers arrays the input will actually accept, because the
                // input says what it accepts.
                let bound = registry_data(&current.params, spec.id);
                let label = bound
                    .and_then(|id| scene.bindable.iter().find(|(held, _)| *held == id))
                    .map(|(id, meta)| format!("d{id} {}", meta.name))
                    .unwrap_or_else(|| {
                        if required {
                            "REQUIRED".into()
                        } else {
                            "none".into()
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    egui::ComboBox::from_id_salt((current.entity, spec.id))
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            let mut any = false;
                            for (id, meta) in &scene.bindable {
                                if spec.kind.accepts(meta).is_err() {
                                    continue;
                                }
                                any = true;
                                let picked = bound == Some(*id);
                                let text =
                                    format!("d{id} {} · {}{:?}", meta.name, meta.dtype, meta.shape);
                                if ui.selectable_label(picked, text).clicked() && !picked {
                                    actions.0.push(UiAction::SetParam(
                                        current.entity,
                                        spec.id,
                                        ParamValue::Data(*id),
                                    ));
                                }
                            }
                            if !any {
                                ui.weak("no uploaded array fits this input");
                            }
                        });
                });
            }
            ParamKind::Choice { options, default } => {
                let chosen = text(&current.params, spec.id, default);
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    for option in options {
                        if ui.selectable_label(chosen == *option, *option).clicked() {
                            actions.0.push(UiAction::SetParam(
                                current.entity,
                                spec.id,
                                ParamValue::Text((*option).to_string()),
                            ));
                        }
                    }
                });
            }
            ParamKind::Bool { default } => {
                let mut value = flag(&current.params, spec.id, default);
                if ui.checkbox(&mut value, spec.label).changed() {
                    actions.0.push(UiAction::SetParam(
                        current.entity,
                        spec.id,
                        ParamValue::Bool(value),
                    ));
                }
            }
        }
    }

}

