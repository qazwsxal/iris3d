//! The Actors tab: everything being drawn, and the controls for one of them.
//!
//! The list is the whole scene rather than just the selected object's actors.
//! Two actors of one object differ only in their settings, and comparing them
//! against a third somewhere else is the reason several exist at all — hiding
//! the rest behind a selection would make that impossible to see. The selected
//! object's group is tinted instead.

use bevy_egui::egui;

use crate::scene::DatasetKind;
use crate::scene::Subset;
use crate::scene::actor::ColorMap;
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
                ui.horizontal(|ui| {
                    let picked = state.selected_actor == Some(actor.entity);
                    if ui
                        .selectable_label(picked, format!("[{}] {}", actor.id, actor.label))
                        .clicked()
                    {
                        actions.0.push(UiAction::SelectActor(actor.entity));
                    }
                    // Worth saying outright: two identical-looking rows over one
                    // object are otherwise indistinguishable when what differs
                    // is which part of the data each draws.
                    if matches!(actor.subset, Subset::Selected { .. }) {
                        ui.label(egui::RichText::new("subset").weak());
                    }
                });
            }
        });
    }

    if !drew_anything {
        ui.weak("Nothing is drawn.");
    }
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
        ui.weak(format!(
            "No registered kind can draw a {} object.",
            row.kind.as_str()
        ));
        return;
    }
    ui.horizontal(|ui| {
        ui.label(format!("add to [{}] {}", row.id, row.name));
        egui::ComboBox::from_id_salt((row.entity, "add"))
            .selected_text("choose a kind")
            .show_ui(ui, |ui| {
                for (id, label) in &row.available {
                    if ui.selectable_label(false, *label).clicked() {
                        actions.0.push(UiAction::AddActor(row.entity, id));
                    }
                }
            });
    });
}

/// One actor: what it is, its parameters, and its colouring.
///
/// The controls are generated from the backend's own `ParamSpec` declarations,
/// so adding an actor kind — or a parameter to an existing one — needs no edit
/// here. A slider's range is the declared range, which is also the range values
/// are clamped to on the way in, so the UI cannot ask for something a client
/// could not.
fn controls(
    ui: &mut egui::Ui,
    scene: &Gathered,
    row: &Row,
    current: &ActorRow,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        ui.heading(current.label);
        ui.weak(format!("of [{}] {}", row.id, row.name));
    });
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
                        if required { "REQUIRED".into() } else { "none".into() }
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
                                let text = format!(
                                    "d{id} {} · {}{:?}",
                                    meta.name, meta.dtype, meta.shape
                                );
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
            ParamKind::Field => {
                // The fields are the source object's, which is why the row this
                // actor was found under comes back with it.
                let chosen = text(&current.params, spec.id, "");
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    egui::ComboBox::from_id_salt((current.entity, spec.id))
                        .selected_text(if chosen.is_empty() { "auto" } else { chosen })
                        .show_ui(ui, |ui| {
                            // Empty means "pick one for me", which is what an
                            // actor starts with.
                            if ui.selectable_label(chosen.is_empty(), "auto").clicked() {
                                actions.0.push(UiAction::SetParam(
                                    current.entity,
                                    spec.id,
                                    ParamValue::Text(String::new()),
                                ));
                            }
                            for field in &row.fields {
                                let picked = chosen == field.name;
                                if ui.selectable_label(picked, &field.name).clicked() && !picked {
                                    actions.0.push(UiAction::SetParam(
                                        current.entity,
                                        spec.id,
                                        ParamValue::Text(field.name.clone()),
                                    ));
                                }
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

    // Colour by. For a molecule, no field means CPK element colouring rather
    // than a flat wash, so name it accordingly.
    let unset = if row.kind == DatasetKind::Molecule {
        "element (CPK)"
    } else {
        "flat"
    };
    ui.horizontal(|ui| {
        ui.label("colour by");
        let selected = current.colour.field.clone().unwrap_or_else(|| unset.into());
        egui::ComboBox::from_id_salt((current.entity, "colour"))
            .selected_text(selected)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current.colour.field.is_none(), unset)
                    .clicked()
                {
                    actions
                        .0
                        .push(UiAction::SetColourField(current.entity, None));
                }
                for field in &row.fields {
                    let picked = current.colour.field.as_deref() == Some(field.name.as_str());
                    // Vector and tensor fields are reduced to magnitude, so say
                    // so rather than implying a direct mapping.
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
                if ui
                    .selectable_label(current.colour.map == map, map.as_str())
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
}
