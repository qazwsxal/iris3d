//! Drawing the window: the menu bar, the panel, and the node canvas.
//!
//! One system, and it only reads. Everything it would change goes onto the
//! action queue instead — see `apply_actions`.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn draw_ui(
    mut contexts: EguiContexts,
    mut state: ResMut<UiState>,
    selection: Res<Selection>,
    mut gizmo: ResMut<GizmoMode>,
    mut actions: ResMut<PendingActions>,
    read: gather::SceneRead,
    arrays: Res<Assets<DataArray>>,
    mut captured: ResMut<PointerCaptured>,
    mut overlays: ResMut<crate::view::viewport::OverlaySettings>,
    pending: Res<Pending>,
    mut graph: ResMut<nodes::NodeGraph>,
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
    let world = gather::gather(&read);

    // How much the panels took, so the 3D camera can be inset to what is left.
    // Without this the scene renders across the whole window and hides behind
    // them.
    let top = egui::Panel::top("menu")
        .show(&mut root, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut state.show_panel, "Panel");
                });
                ui.menu_button("Camera", |ui| {
                    if ui.button("Frame all").clicked() {
                        actions.0.push(UiAction::FrameAll);
                        ui.close();
                    }
                });
                ui.menu_button("Scene", |ui| {
                    if ui.button("Delete all objects").clicked() {
                        for entity in &world.roots {
                            if let Some(row) = world.rows.get(entity) {
                                actions.0.push(UiAction::Delete(row.id));
                            }
                        }
                        ui.close();
                    }
                });
                // What the drag handles do, beside the view toggle rather than
                // in a menu: it is a mode, and a mode that is two clicks away
                // reads as a setting. Only shown in the Scene view, since the
                // node canvas has no handles to switch between.
                if state.view == View::Scene {
                    ui.separator();
                    for (mode, label) in [
                        (GizmoMode::Translate, "Move"),
                        (GizmoMode::Rotate, "Turn"),
                        (GizmoMode::Scale, "Size"),
                    ] {
                        if ui.selectable_label(*gizmo == mode, label).clicked() {
                            *gizmo = mode;
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Right-to-left, so this is the rightmost thing in the bar.
                    let (label, to) = match state.view {
                        View::Scene => ("Nodes", View::Nodes),
                        View::Nodes => ("Scene", View::Scene),
                    };
                    if ui.button(label).clicked() {
                        state.view = to;
                    }
                    ui.separator();
                    // The meshes are counted apart from the arrays because
                    // their vertices are on the GPU rather than in
                    // `Assets<DataArray>` — and because this is where sharing
                    // shows: drawing one ribbon as a lit surface *and* as an
                    // absorbing medium should add an actor and no vertices.
                    let mut summary = format!(
                        "{} objects · {} arrays · {}",
                        world.rows.len(),
                        arrays.len(),
                        human_bytes(world.total_bytes)
                    );
                    if world.meshes > 0 {
                        summary.push_str(&format!(
                            " · {} mesh{}, {} verts",
                            world.meshes,
                            if world.meshes == 1 { "" } else { "es" },
                            world.vertices
                        ));
                    }
                    ui.label(summary);
                });
            });
        })
        .response
        .rect
        .height();

    // The node view takes the window. Drawn before the panel and returning
    // early, so `right` stays zero and the 3D camera is given nothing below —
    // there is no scene on screen to inset it into.
    if state.view == View::Nodes {
        nodes::show(&mut root, &mut graph, &world, &mut actions, &selection);
        // Nothing of the 3D scene is on screen, so the camera is given no
        // viewport at all rather than a sliver behind the canvas.
        if let Ok(mut camera) = cameras.single_mut() {
            camera.viewport = None;
        }
        return Ok(());
    }

    let mut right = 0.0;
    if state.show_panel {
        right = egui::Panel::right("panel")
            .resizable(true)
            .default_size(380.0)
            .show(&mut root, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (Tab::Data, "Data"),
                        (Tab::Filters, "Filters"),
                        (Tab::Actors, "Actors"),
                        (Tab::Scene, "Scene"),
                    ] {
                        if ui.selectable_label(state.tab == tab, label).clicked() {
                            state.tab = tab;
                        }
                    }
                });
                ui.separator();

                // Details before the list. An egui panel claims its space out
                // of what is currently available and leaves the rest to
                // whatever is added afterwards, so a list added first would
                // take the lot and leave the details nothing to sit in.
                let tab = state.tab;
                egui::Panel::bottom("details")
                    .resizable(true)
                    .default_size(280.0)
                    .min_size(60.0)
                    .show(ui, |ui| {
                        // A resizable panel is only as tall as content that
                        // fills it, and a `ScrollArea` shrinks to fit by
                        // default — which collapsed this to a single line
                        // whenever nothing was selected. Refusing to shrink is
                        // what makes the split hold its height.
                        egui::ScrollArea::vertical()
                            .id_salt("details")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                match tab {
                                    Tab::Data => data::details(ui, &world, &selection, &arrays),
                                    Tab::Filters => filters::details(
                                        ui,
                                        &world,
                                        &state,
                                        &selection,
                                        &read.filter_registry,
                                        &mut actions,
                                    ),
                                    Tab::Actors => actors::details(
                                        ui,
                                        &world,
                                        &state,
                                        &selection,
                                        &read.registry,
                                        &mut actions,
                                    ),
                                    Tab::Scene => {
                                        scene::details(ui, &world, &selection, &mut actions)
                                    }
                                }
                                // Whatever the backend last refused. Filters are
                                // the only thing here that can be refused for a
                                // reason worth reading — a binding that does not
                                // fit, a cycle — and the reply that carries it is
                                // otherwise dropped on the floor.
                                if let Some(message) = &pending.error {
                                    ui.separator();
                                    ui.colored_label(ui.visuals().error_fg_color, message);
                                }
                            });
                    });

                egui::ScrollArea::vertical()
                    .id_salt("list")
                    .show(ui, |ui| match tab {
                        Tab::Data => data::list(ui, &world, &selection, &mut actions, &arrays),
                        Tab::Filters => filters::list(ui, &world, &selection, &mut actions),
                        Tab::Actors => actors::list(ui, &world, &selection, &mut actions),
                        Tab::Scene => {
                            scene::list(ui, &world, &selection, &mut actions, &mut overlays)
                        }
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
    let position = UVec2::new(0, (top * scale) as u32);
    let full = UVec2::new(window.physical_width(), window.physical_height());
    let taken = position + UVec2::new((right * scale) as u32, 0);
    camera.viewport = if full.cmpgt(taken).all() {
        Some(Viewport {
            physical_position: position,
            physical_size: full - taken,
            ..default()
        })
    } else {
        // The panel covers everything; a zero-sized viewport would panic.
        None
    };

    Ok(())
}
