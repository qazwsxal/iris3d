//! The Actors tab: everything being drawn, and the controls for one of them.
//!
//! One row per actor, whatever it is drawn under — three columns: what kind it
//! is, what it is, and where it appears. The Scene tab already nests actors
//! under their object, so grouping them by object here as well said the same
//! thing twice and split an actor drawn under three objects into three rows.
//! Flat is the view that tree cannot give: every drawing in the scene, side by
//! side, which is the comparison several of them exist to make.

use bevy_egui::egui;

use crate::model::ParamKind;
use crate::view::select::Selection;

use super::Making;
use super::gather::{ActorRow, Gathered, Row};
use super::params;
use super::{PendingActions, UiAction, UiState};

/// The width of a kind badge, and of the column it sits in.
const BADGE: f32 = 18.0;

/// Colours for the kind badges, indexed by registration order.
///
/// Prototype colouring: a number in a coloured square, so two rows of the same
/// kind are one glance apart before their names are read. When the kinds get
/// real icons this is what they replace.
const PALETTE: [egui::Color32; 6] = [
    egui::Color32::from_rgb(126, 176, 235),
    egui::Color32::from_rgb(235, 165, 120),
    egui::Color32::from_rgb(150, 205, 150),
    egui::Color32::from_rgb(215, 145, 200),
    egui::Color32::from_rgb(230, 210, 130),
    egui::Color32::from_rgb(170, 170, 215),
];

pub fn list(
    ui: &mut egui::Ui,
    scene: &Gathered,
    selection: &Selection,
    actions: &mut PendingActions,
) {
    if scene.actors.is_empty() {
        ui.weak("Nothing is drawn.");
        return;
    }

    // Measured once, before the grid: inside it `available_width` is the width
    // of a cell rather than of the list, so a column asked for it there would
    // grow the grid a little on every frame.
    let full = ui.available_width();
    let names = ((full - BADGE) * 0.5).max(60.0);

    egui::Grid::new("actor-list")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for actor in &scene.actors {
                badge(ui, actor);

                let picked = selection.actor == Some(actor.entity);
                let response = ui
                    .add(
                        egui::Button::selectable(picked, (actor.label, egui::Atom::grow()))
                            .min_size(egui::vec2(names, 0.0)),
                    )
                    .on_hover_text(format!("actor {}", actor.id));
                if response.clicked() {
                    actions.0.push(UiAction::SelectActor(
                        actor.entity,
                        actor.parents.first().map(|(entity, _)| *entity),
                    ));
                }
                response.context_menu(|ui| {
                    if ui.button("Remove").clicked() {
                        actions.0.push(UiAction::RemoveActor(actor.entity));
                        ui.close();
                    }
                });

                parents(ui, actor, actions);
                ui.end_row();
            }
        });
}

/// The kind of an actor, as a coloured number.
fn badge(ui: &mut egui::Ui, actor: &ActorRow) {
    let (rect, response) = ui.allocate_at_least(egui::vec2(BADGE, BADGE), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let colour = PALETTE[actor.badge % PALETTE.len()];
        ui.painter().rect_filled(rect, 4.0, colour);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{}", actor.badge + 1),
            egui::TextStyle::Small.resolve(ui.style()),
            egui::Color32::from_gray(20),
        );
    }
    response.on_hover_text(actor.label);
}

/// The objects an actor is drawn under, as links to them.
///
/// Links rather than text: the question this column answers is "where is this
/// one", and the next thing wanted after the answer is to go there.
fn parents(ui: &mut egui::Ui, actor: &ActorRow, actions: &mut PendingActions) {
    ui.horizontal(|ui| {
        if actor.parents.is_empty() {
            // Nowhere to be drawn, so nothing is on screen. Worth saying
            // outright — otherwise the row looks like every other one.
            ui.weak(egui::RichText::new("not drawn").italics());
            return;
        }
        for (entity, name) in &actor.parents {
            if ui.link(name).clicked() {
                actions
                    .0
                    .push(UiAction::SelectActor(actor.entity, Some(*entity)));
            }
        }
    });
}

pub fn details(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    selection: &Selection,
    registry: &crate::scene::registry::ActorRegistry,
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
    registry: &crate::scene::registry::ActorRegistry,
    actions: &mut PendingActions,
) {
    if draft(ui, scene, state, registry, actions) {
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
        ui.label(format!("add to {}", row.name));
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

/// The form for an actor kind chosen but not yet made, if one is open.
///
/// Every actor kind has at least one required input, so there is nothing worth
/// spawning before they are picked — the command path refuses an unbound actor,
/// and the interface should not be able to make one the wire would reject.
///
/// Returns whether it drew anything, because both places that offer to add an
/// actor — this tab and the scene tree's toolbar — show the form in their own
/// details pane and have to know whether the rest of that pane is wanted.
pub(super) fn draft(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    registry: &crate::scene::registry::ActorRegistry,
    actions: &mut PendingActions,
) -> bool {
    let Some(draft) = &state.draft else {
        return false;
    };
    let Making::Actor(_) = draft.making else {
        return false;
    };
    let Some(kind) = registry.get(draft.kind) else {
        return false;
    };
    params::draft_form(ui, scene, actions, draft, kind.label, kind.params, |spec| {
        // Geometry never comes from an upload, so an empty picker here is a
        // missing filter rather than missing data.
        matches!(spec.kind, ParamKind::Geometry { .. }).then_some("assemble…")
    });
    true
}

/// One actor: what it is and its parameters.
///
/// The controls are generated from the backend's own `ParamSpec` declarations,
/// so adding an actor kind — or a parameter to an existing one — needs no edit
/// here. A slider's range is the declared range, which is also the range values
/// are clamped to on the way in, so the UI cannot ask for something a client
/// could not.
pub(super) fn controls(
    ui: &mut egui::Ui,
    scene: &Gathered,
    row: Option<&Row>,
    current: &ActorRow,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        ui.heading(current.label);
        match row {
            Some(row) => ui.weak(format!("of {}", row.name)),
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
