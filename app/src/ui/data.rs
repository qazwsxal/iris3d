//! The Data tab: the arrays in memory, and what they mean.
//!
//! Two listings rather than one, because the same bytes answer two questions.
//! An array is a buffer with a dtype and a size, which is what the inventory
//! shows; it is also, often, a named field over a dataset, which is what the
//! colour pickers and the `Field` parameters offer. Grids make the split
//! obvious — every one of their buffers is a field and none is geometry.

use bevy::asset::Assets;
use bevy_egui::egui;

use crate::scene::DataArray;

use super::gather::Gathered;
use super::{PendingActions, UiAction, UiState, human_bytes};

pub fn list(
    ui: &mut egui::Ui,
    scene: &Gathered,
    state: &UiState,
    actions: &mut PendingActions,
    arrays: &Assets<DataArray>,
) {
    ui.weak(format!(
        "{} in memory · {}",
        arrays.len(),
        human_bytes(scene.total_bytes)
    ));
    ui.add_space(4.0);

    let mut listing: Vec<_> = arrays.iter().collect();
    listing.sort_by_key(|(id, _)| {
        scene
            .owners
            .get(id)
            .map(|owner| owner.object)
            .unwrap_or(u64::MAX)
    });

    egui::Grid::new("array-grid")
        .num_columns(4)
        .striped(true)
        .show(ui, |ui| {
            ui.strong("owner");
            ui.strong("name");
            ui.strong("type");
            ui.strong("size");
            ui.end_row();

            for (id, array) in listing {
                let owner = scene.owners.get(&id);
                let held = scene.held.get(&id);
                // Three states, and they are genuinely different: an array an
                // object holds, an array uploaded on its own and waiting to be
                // bound, and an array nothing refers to at all.
                match (owner, held) {
                    (Some(owner), _) => {
                        ui.monospace(format!("{}", owner.object));
                    }
                    (None, Some((handle, _))) => {
                        ui.monospace(format!("d{handle}"));
                    }
                    (None, None) => {
                        ui.weak("—");
                    }
                }
                // The name is the click target: a whole grid row is not one
                // widget in egui, so something in it has to carry the click.
                let name = owner
                    .map(|owner| owner.name.as_str())
                    .or_else(|| held.map(|(_, name)| name.as_str()));
                if ui
                    .selectable_label(
                        state.selected_array == Some(id),
                        name.unwrap_or("<unreferenced>"),
                    )
                    .clicked()
                {
                    actions.0.push(UiAction::SelectArray(id));
                }
                ui.monospace(format!(
                    "{}{:?}",
                    array.dtype,
                    array.shape.iter().collect::<Vec<_>>()
                ));
                ui.monospace(human_bytes(array.data.len() as u64));
                ui.end_row();
            }
        });

    ui.add_space(8.0);
    ui.separator();
    ui.strong("Fields");

    let mut any = false;
    egui::Grid::new("field-grid")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for object in &scene.ordered {
                let Some(row) = scene.rows.get(object) else {
                    continue;
                };
                for field in &row.fields {
                    any = true;
                    ui.monospace(format!("{}", row.id));
                    ui.label(&field.name);
                    ui.weak(field.kind);
                    ui.end_row();
                }
            }
        });
    if !any {
        ui.weak("Nothing loaded carries a field.");
    }
}

pub fn details(ui: &mut egui::Ui, scene: &Gathered, state: &UiState, arrays: &Assets<DataArray>) {
    let Some(id) = state.selected_array else {
        ui.weak("Select an array.");
        return;
    };
    // The selection is by asset id, and the object holding it can be deleted
    // while the details are on screen.
    let Some(array) = arrays.get(id) else {
        ui.weak("That array is gone.");
        return;
    };
    let owner = scene.owners.get(&id);

    ui.heading(
        owner
            .map(|owner| owner.name.as_str())
            .unwrap_or("<unreferenced>"),
    );
    ui.separator();

    egui::Grid::new("array-details")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("type");
            ui.monospace(format!("{}", array.dtype));
            ui.end_row();
            ui.label("shape");
            ui.monospace(format!("{:?}", array.shape.iter().collect::<Vec<_>>()));
            ui.end_row();
            ui.label("size");
            ui.monospace(human_bytes(array.data.len() as u64));
            ui.end_row();
            ui.label("owner");
            match owner {
                Some(owner) => {
                    ui.monospace(format!("object {}", owner.object));
                }
                None => {
                    ui.weak("nothing holds it");
                }
            }
            ui.end_row();
        });

    let Some(owner) = owner else { return };
    if owner.fields.is_empty() {
        return;
    }
    ui.separator();
    ui.label("read as");
    for field in &owner.fields {
        ui.monospace(field);
    }
}
