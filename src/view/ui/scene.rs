//! The Scene tab: the object tree, and what is drawn under each object.
//!
//! One tree, laid out the way a 3D application lays out an outliner: a row per
//! thing, its children nested under a disclosure triangle, and the toggles that
//! belong to a row on that row rather than in a panel underneath. An object's
//! actors are children here, because "what draws this" is part of where a thing
//! sits — reaching them through a second tab meant clicking twice to answer a
//! question the tree was already showing half of.
//!
//! What a row cannot fit goes on its context menu. Framing and deletion are one
//! object at a time and are asked for rarely, so they cost a right-click rather
//! than permanent width.

use bevy_egui::egui;

use crate::scene::registry::ActorRegistry;
use crate::view::viewport::OverlaySettings;

use super::gather::{ActorRow, Gathered, Row};
use super::{PendingActions, UiAction, UiState};
use crate::view::select::Selection;

/// Width kept clear at the right end of a row for its toggles.
///
/// Reserved rather than measured: the label is a full-width button so that the
/// highlight covers the row, which means the space the eye needs has to come
/// out of the label's width before the label is added.
const TOGGLES: f32 = 26.0;

pub fn list(
    ui: &mut egui::Ui,
    scene: &Gathered,
    selection: &Selection,
    actions: &mut PendingActions,
    overlays: &mut OverlaySettings,
) {
    toolbar(ui, scene, selection, actions, overlays);
    ui.separator();

    if scene.roots.is_empty() && scene.detached.is_empty() {
        ui.weak("Nothing loaded. Upload over gRPC.");
    }
    for root in &scene.roots {
        node(ui, *root, scene, selection, actions);
    }

    // Actors drawn under nothing, usually because the last object they were
    // under was deleted. They are in no object's branch, so without a branch of
    // their own there would be nothing on screen to say they still exist.
    if !scene.detached.is_empty() {
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id("detached"),
            false,
        )
        .show_header(ui, |ui| {
            ui.label(egui::RichText::new("Detached — not drawn").weak().italics());
        })
        .body(|ui| {
            for actor in &scene.detached {
                actor_row(ui, actor, None, selection, actions);
            }
        });
    }
}

/// The row above the tree: what can be added, and what is drawn over the scene.
///
/// Both are about the tree as a whole rather than about any one row, which is
/// what puts them here instead of on a row or in the details below.
fn toolbar(
    ui: &mut egui::Ui,
    scene: &Gathered,
    selection: &Selection,
    actions: &mut PendingActions,
    overlays: &mut OverlaySettings,
) {
    ui.horizontal(|ui| {
        let object = selection.object.and_then(|entity| scene.rows.get(&entity));
        ui.menu_button("+", |ui| {
            if ui.button("New object").clicked() {
                actions.0.push(UiAction::AddObject {
                    name: fresh_name(scene),
                    inside: None,
                });
                ui.close();
            }
            let Some(row) = object else {
                return;
            };
            if ui
                .button(format!("New object inside {}", row.name))
                .clicked()
            {
                actions.0.push(UiAction::AddObject {
                    name: fresh_name(scene),
                    inside: Some(row.entity),
                });
                ui.close();
            }
            ui.separator();
            ui.weak(format!("draw {} as", row.name));
            if row.available.is_empty() {
                ui.weak("This build has no registered actor kinds.");
            }
            for kind in &row.available {
                if ui.button(kind.label).clicked() {
                    actions.0.push(UiAction::AddActor(row.entity, kind.id));
                    ui.close();
                }
            }
        })
        .response
        .on_hover_text("Add an object, or draw the selected one another way");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.menu_button("Display", |ui| {
                ui.checkbox(&mut overlays.grid, "Ground grid");
                ui.checkbox(&mut overlays.world_axes, "World axes");
                ui.checkbox(&mut overlays.selection, "Selection outline");
                ui.checkbox(&mut overlays.all_bounds, "All bounds");
            });
        });
    });
}

/// A name no object in the scene has yet.
///
/// "object", then "object 2". Numbered rather than left to collide because the
/// tree names things and no longer shows their handles — two rows reading the
/// same is two rows that cannot be told apart.
fn fresh_name(scene: &Gathered) -> String {
    let taken = |name: &str| scene.rows.values().any(|row| row.name == name);
    if !taken("object") {
        return "object".to_string();
    }
    (2..)
        .map(|n| format!("object {n}"))
        .find(|name| !taken(name))
        .expect("an unbounded range contains a free name")
}

/// One object, its nested objects, and the actors drawn under it.
///
/// A row with nothing under it gets a plain row indented to where a triangle
/// would have put it, so the names still line up without the tree filling with
/// disclosure triangles that expand into nothing.
fn node(
    ui: &mut egui::Ui,
    entity: bevy::prelude::Entity,
    scene: &Gathered,
    selection: &Selection,
    actions: &mut PendingActions,
) {
    let Some(row) = scene.rows.get(&entity) else {
        return;
    };

    if row.children.is_empty() && row.actors.is_empty() {
        ui.horizontal(|ui| {
            indent(ui);
            object_row(ui, row, selection, actions);
        });
        return;
    }

    egui::collapsing_header::CollapsingState::load_with_default_open(
        ui.ctx(),
        ui.make_persistent_id(entity),
        true,
    )
    .show_header(ui, |ui| {
        object_row(ui, row, selection, actions);
    })
    .body(|ui| {
        for child in &row.children {
            node(ui, *child, scene, selection, actions);
        }
        for actor in &row.actors {
            actor_row(ui, actor, Some(row), selection, actions);
        }
    });
}

/// The space a disclosure triangle takes, for rows that have none.
fn indent(ui: &mut egui::Ui) {
    ui.add_space(ui.spacing().icon_width + ui.spacing().icon_spacing);
}

/// The label and the eye of one object.
///
/// Called from inside a horizontal layout — either the one a collapsing header
/// makes for its own row, or the one [`node`] makes for a leaf.
fn object_row(ui: &mut egui::Ui, row: &Row, selection: &Selection, actions: &mut PendingActions) {
    let selected = selection.object == Some(row.entity) && selection.actor.is_none();
    let response = name(ui, selected, dimmed(ui, &row.name, row.shown))
        .on_hover_text(format!("object {}", row.id));
    if response.clicked() {
        actions.0.push(UiAction::Select(row.entity));
    }
    if response.double_clicked() {
        actions.0.push(UiAction::Frame(row.entity));
    }
    response.context_menu(|ui| {
        if ui.button("Frame").clicked() {
            actions.0.push(UiAction::Frame(row.entity));
            ui.close();
        }
        if ui
            .button(if row.visible { "Hide" } else { "Show" })
            .clicked()
        {
            actions.0.push(UiAction::ToggleVisibility(row.entity));
            ui.close();
        }
        ui.separator();
        if ui.button("Delete").clicked() {
            actions.0.push(UiAction::Delete(row.id));
            ui.close();
        }
    });

    if eye(ui, row).clicked() {
        actions.0.push(UiAction::ToggleVisibility(row.entity));
    }
}

/// One actor as a child row.
///
/// `under` is the object whose branch this row is in, which is what the click
/// selects alongside the actor. One actor drawn under several objects has a row
/// in each of their branches, and they differ only in this.
fn actor_row(
    ui: &mut egui::Ui,
    actor: &ActorRow,
    under: Option<&Row>,
    selection: &Selection,
    actions: &mut PendingActions,
) {
    ui.horizontal(|ui| {
        indent(ui);
        let selected = selection.actor == Some(actor.entity);
        // An actor is drawn where its object is drawn, so it follows the row
        // above it rather than carrying a flag of its own.
        let shown = under.is_some_and(|row| row.shown);
        let response = name(ui, selected, dimmed(ui, actor.label, shown))
            .on_hover_text(format!("actor {}", actor.id));
        if response.clicked() {
            actions
                .0
                .push(UiAction::SelectActor(actor.entity, under.map(|r| r.entity)));
        }
        response.context_menu(|ui| {
            if ui.button("Remove").clicked() {
                actions.0.push(UiAction::RemoveActor(actor.entity));
                ui.close();
            }
        });
        // One actor, several places: worth saying on the row, because removing
        // it here removes the drawing itself rather than this one placement.
        if actor.places > 1 {
            ui.weak(format!("x{}", actor.places))
                .on_hover_text("Drawn under several objects");
        }
    });
}

/// The clickable name of a row.
///
/// A button grown to the width of the row rather than a label sized to its
/// text, so the highlight is the row and not the word — clicking the empty
/// space beside a name is how an outliner is expected to behave. `TOGGLES` is
/// held back for whatever the row puts at its right end.
/// A row's text, faded when what it stands for is not on screen.
///
/// The colour is named outright rather than left to `RichText::weak`, because
/// a row's text is an atom inside a button and a button supplies a colour of its
/// own for atoms that ask for none. Saying which colour settles it either way.
fn dimmed(ui: &egui::Ui, text: &str, shown: bool) -> egui::RichText {
    let label = egui::RichText::new(text);
    if shown {
        label
    } else {
        label.color(ui.visuals().weak_text_color())
    }
}

fn name(ui: &mut egui::Ui, selected: bool, label: egui::RichText) -> egui::Response {
    let width = (ui.available_width() - TOGGLES).max(48.0);
    ui.add(
        egui::Button::selectable(selected, (label, egui::Atom::grow()))
            .min_size(egui::vec2(width, 0.0)),
    )
}

/// The visibility toggle at the right end of an object's row.
///
/// The eye reports what is on screen, so it dims for an object hidden with its
/// parent as well as for one hidden on its own. Clicking still toggles this
/// object's own flag — which is why the two cases say different things on
/// hover: one of them will not put anything back on screen by itself.
fn eye(ui: &mut egui::Ui, row: &Row) -> egui::Response {
    let glyph = dimmed(ui, "👁", row.shown);
    let response = ui.add(egui::Button::new(glyph).frame(false).small());
    match (row.visible, row.shown) {
        (_, true) => response.on_hover_text("Hide"),
        (false, false) => response.on_hover_text("Show"),
        (true, false) => response.on_hover_text("Hidden with its parent"),
    }
}

/// What the selected row is, below the tree.
///
/// An actor's row selects the actor, so the space under the tree shows that
/// actor's controls — the same ones the Actors tab draws, from the same
/// declarations. An object has no settings of its own yet, so it gets its counts
/// and nothing more; the buttons that used to be here are on the rows and their
/// context menus now.
pub fn details(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    selection: &Selection,
    registry: &ActorRegistry,
    actions: &mut PendingActions,
) {
    // A kind chosen from the toolbar's + but not yet made: every actor kind has
    // a required input, so this is where those are bound before it is spawned.
    if super::actors::draft(ui, scene, state, registry, actions) {
        return;
    }

    if let Some((row, actor)) = selection.actor.and_then(|entity| scene.actor(entity)) {
        super::actors::controls(ui, scene, row, actor, actions);
        return;
    }

    let Some(row) = selection.object.and_then(|entity| scene.rows.get(&entity)) else {
        ui.weak("Select a row.");
        return;
    };

    ui.heading(&row.name);
    ui.weak(format!("object {}", row.id));
    ui.separator();
    egui::Grid::new("object-details")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("actors");
            ui.monospace(format!("{}", row.actors.len()));
            ui.end_row();
            ui.label("children");
            ui.monospace(format!("{}", row.children.len()));
            ui.end_row();
        });
}
