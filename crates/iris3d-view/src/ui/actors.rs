//! The Actors tab: everything being drawn, and the controls for one of them.
//!
//! The list is the whole scene rather than just the selected object's actors.
//! Two actors of one object differ only in their settings, and comparing them
//! against a third somewhere else is the reason several exist at all — hiding
//! the rest behind a selection would make that impossible to see. The selected
//! object's group is tinted instead.

use bevy_egui::egui;

use crate::select::Selection;
use iris3d_model::ParamKind;

use super::Making;
use super::gather::{ActorRow, Gathered, Row};
use super::params;
use super::{PendingActions, UiAction, UiState};

pub fn list(
    ui: &mut egui::Ui,
    scene: &Gathered,
    selection: &Selection,
    actions: &mut PendingActions,
) {
    let mut drew_anything = false;

    for object in &scene.ordered {
        let Some(row) = scene.rows.get(object) else {
            continue;
        };
        if row.actors.is_empty() {
            continue;
        }
        drew_anything = true;

        let highlighted = selection.object == Some(row.entity);
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
                entry(ui, actor, Some(row), selection, actions);
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
                entry(ui, actor, None, selection, actions);
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
    selection: &Selection,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        let picked = selection.actor == Some(actor.entity);
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
    });
}

pub fn details(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    selection: &Selection,
    registry: &iris3d_scene::registry::ActorRegistry,
    actions: &mut PendingActions,
) {
    match selection.actor.and_then(|entity| scene.actor(entity)) {
        Some((row, actor)) => {
            controls(ui, scene, row, actor, actions);
            ui.separator();
        }
        None => {
            ui.weak("Select an actor.");
            ui.separator();
        }
    }
    add(ui, scene, state, selection, registry, actions);
}

/// The "draw this another way" row.
///
/// Adding rather than replacing: an object may be drawn several ways at once,
/// and each way is its own entity with its own settings.
fn add(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    selection: &Selection,
    registry: &iris3d_scene::registry::ActorRegistry,
    actions: &mut PendingActions,
) {
    // A kind chosen but not yet made. Every actor kind has at least one required
    // input, so there is nothing worth spawning before they are picked — the
    // command path refuses an unbound actor, and the interface should not be
    // able to make one the wire would reject.
    if let Some(draft) = &state.draft
        && let Making::Actor(_) = draft.making
        && let Some(kind) = registry.get(draft.kind)
    {
        params::draft_form(ui, scene, actions, draft, kind.label, kind.params, |spec| {
            // Geometry never comes from an upload, so an empty picker here is a
            // missing filter rather than missing data.
            matches!(spec.kind, ParamKind::Geometry { .. }).then_some("assemble…")
        });
        return;
    }

    let Some(row) = selection.object.and_then(|entity| scene.rows.get(&entity)) else {
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
        if ui.small_button("remove").clicked() {
            actions.0.push(UiAction::RemoveActor(current.entity));
        }
    });
    ui.separator();

    // An actor's geometry can only come from a filter, so an empty picker here
    // is not a missing upload — it is a missing `geometry` filter. Offering to
    // make one is the difference between this panel working and dead-ending;
    // asking a person to know that assembly is a filter's job is the tool
    // exposing its own plumbing.
    let edits = params::controls(
        ui,
        scene,
        current.specs,
        &current.params,
        current.entity,
        |spec| matches!(spec.kind, ParamKind::Geometry { .. }).then_some("assemble…"),
    );
    for (id, value) in edits.set {
        actions
            .0
            .push(UiAction::SetParam(current.entity, id, value));
    }
    if let Some(input) = edits.offered {
        actions.0.push(UiAction::OfferFilter {
            kind: "geometry",
            then: Some(("geometry", super::Target::Actor(current.entity, input))),
        });
    }
}
