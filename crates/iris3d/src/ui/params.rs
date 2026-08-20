//! Controls generated from a kind's own `ParamSpec` declarations.
//!
//! Written once and used by both tabs, because a filter kind declares its
//! parameters exactly as an actor kind does — deliberately, so that the split
//! between deriving data and drawing it costs the interface nothing. A new kind
//! of either sort gets its controls for free, and a new *parameter* on an
//! existing kind does too.
//!
//! Nothing here knows what an actor or a filter is. It is handed a declaration,
//! the values currently against it, and somewhere to report edits; what those
//! edits mean is the caller's business, and is the only thing that differs
//! between the two.
//!
//! A slider's range is the declared range, which is also what values are clamped
//! to on the way in through `sanitise`. So the interface cannot ask for anything
//! a scripted client could not, and a declaration with a useless range gives a
//! useless control — the fix for that belongs in the declaration.

use std::hash::Hash;

use bevy_egui::egui;

use iris3d_model::{ParamKind, ParamMap, ParamSpec, ParamValue, flag, float, text};
use iris3d_model::{data as bound_handle, vector as param_vector};

use super::gather::Gathered;
use super::{Draft, Making, PendingActions, Target, UiAction};

/// What the user did to a set of controls in one frame.
#[derive(Default)]
pub struct Edits {
    /// Parameters moved, as the kind's own ids.
    pub set: Vec<(&'static str, ParamValue)>,
    /// The input whose offer button was pressed.
    ///
    /// Reported rather than acted on: filling an input means creating whichever
    /// filter produces that sort of data, and only the caller knows which that
    /// is or what to do with it afterwards.
    pub offered: Option<&'static str>,
}

/// Draws one control per declared parameter and collects what changed.
///
/// `salt` separates one set of controls from another in egui's id space; the
/// owning entity is the natural thing to pass. `offer` is asked, for each
/// *unbound* input, whether to show a button beside the picker and what to call
/// it — returning `None` everywhere means no offers at all.
pub fn controls(
    ui: &mut egui::Ui,
    scene: &Gathered,
    specs: &'static [ParamSpec],
    params: &ParamMap,
    salt: impl Hash + Copy + std::fmt::Debug,
    offer: impl Fn(&ParamSpec) -> Option<&'static str>,
) -> Edits {
    controls_where(ui, scene, specs, params, salt, offer, |_| true)
}

/// [`controls`], over the declared parameters `include` accepts.
///
/// Exists for the node canvas, which draws a kind's *settings* in the node body
/// and its *inputs* as pins — the same declaration split across two places
/// rather than shown twice. Panels take the whole set and so call [`controls`].
///
/// The predicate rather than a filtered slice: `specs` is `&'static`, so
/// narrowing it would mean allocating a new one every frame to say something the
/// caller already knows how to test.
pub fn controls_where(
    ui: &mut egui::Ui,
    scene: &Gathered,
    specs: &'static [ParamSpec],
    params: &ParamMap,
    salt: impl Hash + Copy + std::fmt::Debug,
    offer: impl Fn(&ParamSpec) -> Option<&'static str>,
    include: impl Fn(&ParamSpec) -> bool,
) -> Edits {
    let mut edits = Edits::default();

    for spec in specs.iter().filter(|spec| include(spec)) {
        match spec.kind {
            ParamKind::Float {
                default,
                min,
                max,
                logarithmic,
            } => {
                let mut value = float(params, spec.id, default);
                if ui
                    .add(
                        egui::Slider::new(&mut value, min..=max)
                            .logarithmic(logarithmic)
                            .text(spec.label),
                    )
                    .changed()
                {
                    edits.set.push((spec.id, ParamValue::Float(value)));
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
                let mut values = param_vector(params, spec.id, components);
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
                    edits.set.push((spec.id, ParamValue::Vector(values)));
                }
            }
            ParamKind::Array { required, .. } | ParamKind::Geometry { required } => {
                // Generated from the same declaration as every other control,
                // which is the point of bindings being parameters: the picker
                // only offers what the input will actually accept, because the
                // input says what it accepts. Arrays and meshes go through the
                // one control for the same reason — they are one handle space,
                // and `accepts` is what separates them.
                let bound = bound_handle(params, spec.id);
                let label = bound
                    .map(|handle| scene.describe_handle(handle))
                    .unwrap_or_else(|| {
                        if required {
                            "REQUIRED".into()
                        } else {
                            "none".into()
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    egui::ComboBox::from_id_salt((salt, spec.id))
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            let mut any = false;
                            for (id, meta) in &scene.bindable {
                                if spec.kind.accepts(meta.as_held()).is_err() {
                                    continue;
                                }
                                any = true;
                                let picked = bound == Some(*id);
                                let text =
                                    format!("{} · {}", scene.describe_handle(*id), meta.describe());
                                if ui.selectable_label(picked, text).clicked() && !picked {
                                    edits.set.push((spec.id, ParamValue::Data(*id)));
                                }
                            }
                            if !any {
                                ui.weak("nothing held fits this input");
                            }
                        });
                    // Only where there is nothing to pick yet. An offer beside a
                    // full picker is noise; beside an empty one it is the
                    // difference between a dead end and a next step.
                    if bound.is_none()
                        && let Some(label) = offer(spec)
                        && ui.small_button(label).clicked()
                    {
                        edits.offered = Some(spec.id);
                    }
                });
            }
            ParamKind::Choice { options, default } => {
                let chosen = text(params, spec.id, default);
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    for option in options {
                        if ui.selectable_label(chosen == *option, *option).clicked() {
                            edits
                                .set
                                .push((spec.id, ParamValue::Text((*option).to_string())));
                        }
                    }
                });
            }
            ParamKind::Bool { default } => {
                let mut value = flag(params, spec.id, default);
                if ui.checkbox(&mut value, spec.label).changed() {
                    edits.set.push((spec.id, ParamValue::Bool(value)));
                }
            }
            ParamKind::Text { default } => {
                let mut value = text(params, spec.id, default).to_string();
                ui.horizontal(|ui| {
                    ui.label(spec.label);
                    // On losing focus or pressing enter rather than per
                    // keystroke: every edit is a `SetFilter` that marks the
                    // filter stale, and re-running a match on each letter of
                    // "HOH" would spend three runs to answer the first two
                    // typos.
                    let box_ = egui::TextEdit::singleline(&mut value).desired_width(120.0);
                    if ui.add(box_).lost_focus() {
                        edits.set.push((spec.id, ParamValue::Text(value.clone())));
                    }
                });
            }
        }
    }

    edits
}

/// The first required input with nothing bound to it.
///
/// Neither an actor nor a filter can be created until every one of these is
/// filled — both check bindings before anything is spawned — so the create
/// button needs to name the one still missing rather than simply refusing.
pub fn missing_input(specs: &'static [ParamSpec], params: &ParamMap) -> Option<&'static str> {
    specs
        .iter()
        .filter(|spec| spec.kind.is_required())
        .find(|spec| bound_handle(params, spec.id).is_none())
        .map(|spec| spec.label)
}

/// The staged create form, for an actor or a filter alike.
///
/// One form for both because the reason it exists is the same for both: the
/// backend refuses anything whose required inputs are unbound, so the bindings
/// have to be gathered somewhere before there is a thing to hang them on.
pub fn draft_form(
    ui: &mut egui::Ui,
    scene: &Gathered,
    actions: &mut PendingActions,
    draft: &Draft,
    label: &str,
    specs: &'static [ParamSpec],
    offer: impl Fn(&ParamSpec) -> Option<&'static str>,
) {
    ui.horizontal(|ui| {
        ui.heading(format!("new {label}"));
        if ui.small_button("cancel").clicked() {
            actions.0.push(UiAction::CancelDraft);
        }
    });
    // Where it is going, when it was opened to fill something in. Worth saying:
    // it is the reason the form appeared, and the wiring afterwards happens
    // without another click, so nothing else would account for it.
    if let Making::Filter(Some((output, target))) = &draft.making {
        let into = match target {
            Target::Actor(_, input) => format!("the selected actor's {input}"),
            Target::Filter(id, input) => format!("[{id}]'s {input}"),
            Target::NewActor { kind, input, .. } => format!("a new {kind}'s {input}"),
        };
        ui.weak(format!("its {output} will be bound to {into}"));
    }

    let edits = controls(ui, scene, specs, &draft.params, "draft", offer);
    for (input, value) in edits.set {
        actions.0.push(UiAction::SetDraftParam(input, value));
    }
    if let Some(input) = edits.offered {
        actions.0.push(UiAction::OfferFilter {
            kind: "geometry",
            then: Some((
                "geometry",
                match draft.making {
                    // The actor cannot be made before the thing it requires, so
                    // the draft is spent into the offer rather than kept.
                    Making::Actor(object) => Target::NewActor {
                        object,
                        kind: draft.kind,
                        input,
                    },
                    Making::Filter(_) => continue_unreachable(),
                },
            )),
        });
    }

    ui.separator();
    match missing_input(specs, &draft.params) {
        Some(missing) => {
            ui.add_enabled(false, egui::Button::new("create"));
            ui.weak(format!("needs {missing}"));
        }
        None => {
            if ui.button("create").clicked() {
                actions.0.push(UiAction::Create {
                    kind: draft.kind,
                    params: draft.params.clone(),
                    making: draft.making,
                });
            }
        }
    }
}

/// A filter draft offers nothing, so this arm cannot be reached.
///
/// Spelled out rather than `unreachable!` because a panic in a draw system takes
/// the window with it, and a wrong offer is not worth that.
fn continue_unreachable() -> Target {
    Target::Filter(u64::MAX, "")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Dtype;

    const SPECS: &[ParamSpec] = &[
        ParamSpec {
            id: "level",
            label: "level",
            kind: ParamKind::Float {
                default: 0.5,
                min: 0.0,
                max: 1.0,
                logarithmic: false,
            },
        },
        ParamSpec {
            id: "step",
            label: "step",
            kind: ParamKind::Vector {
                components: 3,
                default: &[1.0, 1.0, 1.0],
                min: 1.0,
                max: 32.0,
                integral: true,
            },
        },
        ParamSpec {
            id: "map",
            label: "map",
            kind: ParamKind::Choice {
                options: &["viridis", "grayscale"],
                default: "viridis",
            },
        },
        ParamSpec {
            id: "field",
            label: "field",
            kind: ParamKind::Array {
                dtypes: &[Dtype::Float32],
                shape: &[0],
                required: true,
                structural: true,
            },
        },
    ];

    /// Drawing the controls, with nobody touching them, changes nothing.
    ///
    /// Worth pinning because the failure is invisible. A widget that writes its
    /// value back on every frame sends a `SetFilterParam` every frame, which
    /// looks like a value drifting on its own — and the value it drifts *to*
    /// looks plausible, so it reads as the data being odd rather than the
    /// interface editing it. Each control is generated, so one such widget would
    /// do this to every kind at once.
    #[test]
    fn drawing_the_controls_edits_nothing() {
        let scene = Gathered::default();
        let mut params = ParamMap::default();
        params.insert("level".into(), ParamValue::Float(0.5));
        params.insert("step".into(), ParamValue::Vector(vec![1.0, 1.0, 1.0]));
        params.insert("map".into(), ParamValue::Text("viridis".into()));

        let mut edited = Vec::new();
        egui::__run_test_ui(|ui| {
            let edits = controls(ui, &scene, SPECS, &params, "test", |_| None);
            edited.extend(edits.set.iter().map(|(id, _)| *id));
            assert!(edits.offered.is_none());
        });
        assert!(
            edited.is_empty(),
            "drawing alone reported edits to {edited:?}"
        );
    }
}
