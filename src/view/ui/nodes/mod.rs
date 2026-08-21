//! The node view: the scene as a graph, wired together by hand.
//!
//! The tabbed panel lists each kind of thing separately and says in words what
//! is connected to what — `from [3] colour map` on an input, `→ [9] surface` on
//! an output. That works, and it is what makes a chain followable at all, but a
//! list can only ever describe a graph. This draws it.
//!
//! **The scene is the truth; this is a projection of it.** Node positions are
//! the one thing here that the world does not know, so the graph persists in a
//! resource and is *reconciled* against [`Gathered`] each frame rather than
//! rebuilt — nodes appear and disappear to match, and keep where they were put.
//! Wires are the opposite: they are rebuilt from the scene every frame, so a
//! wire the user drags does not connect anything directly. It emits the same
//! [`UiAction`] the panel's pickers emit, the scene changes, and the wire shows
//! up next frame because the binding now exists. One source of truth, and the
//! same validation a scripted client gets.
//!
//! Three relations are drawn, and they are genuinely different things:
//!
//! - **data flow**, an output handle into an input parameter;
//! - **placement**, an actor into the object it is drawn under;
//! - **nesting**, an object into its parent object.
//!
//! Only the first is editable here so far. The other two are drawn because
//! leaving them out makes the picture lie — an actor drawn nowhere is the
//! panel's "Detached — not drawn" state, and on a canvas it is simply a node
//! with no placement wire, which explains itself.

use bevy::asset::AssetId;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32};
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::model::data as bound_handle;
use crate::model::{ParamKind, ParamValue};
use crate::scene::DataArray;
use crate::view::select::Selection;

use super::gather::Gathered;
use super::{PendingActions, UiAction, params};

/// One thing on the canvas, named by what identifies it in the scene.
///
/// Deliberately just a key: everything drawn about a node — its label, its
/// parameters, what its outputs are called — is looked up in [`Gathered`] while
/// drawing. Copying that into the node would mean two versions of the same
/// facts, and the stale one would be the one on screen.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    /// A data handle: an upload, or a filter's output.
    Data(u64),
    Filter(u64),
    Actor(Entity),
    Object(Entity),
}

/// The canvas, and the mapping that lets it be reconciled rather than rebuilt.
#[derive(Resource, Default)]
pub struct NodeGraph {
    snarl: Snarl<Node>,
    placed: HashMap<Node, NodeId>,
}

/// An object's two input pins: what is drawn under it, and what is nested in it.
const PLACED_HERE: usize = 0;
const NESTED_HERE: usize = 1;

/// Colour by relation, so the three kinds of wire never read as one.
const DATA: Color32 = Color32::from_rgb(120, 170, 255);
const PLACEMENT: Color32 = Color32::from_rgb(235, 170, 80);
const NESTING: Color32 = Color32::from_rgb(150, 150, 150);

/// Where a newly seen node is put, by kind.
///
/// Left to right in the order the data moves: uploads, then what derives from
/// them, then what draws it, then what it is drawn under. A real layout would
/// follow the edges, but a column per kind already puts most chains roughly in
/// order and costs nothing to compute — and the moment a node is dragged, this
/// never applies to it again.
///
/// The spacing has to clear the widest node rather than look tidy on paper: a
/// filter's rows are its parameter labels, and `geometry`'s "vertices to keep"
/// is wider than the 280 this first used, so the columns overlapped.
const COLUMN: f32 = 420.0;

/// How wide a node's controls are allowed to be.
///
/// Set here rather than left to egui, because a slider expands to fill whatever
/// it is offered and a node body is offered the canvas. Narrow enough that a
/// filter with six parameters is still a node rather than a panel.
const BODY_WIDTH: f32 = 220.0;

/// Above this many controls, a node's body gains a header that can fold it away.
///
/// Not a limit on what is shown — everything is shown — only on which kinds are
/// worth offering the fold to. A node with two controls has nothing to gain from
/// a row spent saying so.
const INLINE_SETTINGS: usize = 3;

mod layout;
mod viewer;

use layout::{inputs_of, is_setting, settings_of};
use viewer::Viewer;

pub fn show(
    ui: &mut egui::Ui,
    graph: &mut NodeGraph,
    scene: &Gathered,
    actions: &mut PendingActions,
    selection: &Selection,
) {
    graph.reconcile(scene);
    graph.rewire(scene);

    let mut viewer = Viewer {
        scene,
        actions,
        selection,
    };
    SnarlWidget::new()
        .id(egui::Id::new("iris3d-nodes"))
        .style(SnarlStyle::new())
        .show(&mut graph.snarl, &mut viewer, ui);
}

/// An input row's label, and whether anything is bound to it.
///
/// A required input with no wire is the one thing worth shouting about: it is
/// why a filter produces nothing and why an actor draws nothing, and on a canvas
/// the absence of a wire is easy to miss among the ones that are there.
fn label_for(ui: &mut egui::Ui, spec: Option<&'static crate::model::ParamSpec>, pin: &InPin) {
    let Some(spec) = spec else {
        return;
    };
    if pin.remotes.is_empty() && spec.kind.is_required() {
        ui.colored_label(ui.visuals().error_fg_color, spec.label);
    } else {
        ui.label(spec.label);
    }
}

/// Every actor in the scene with the entity that names it, attached or not.
///
/// Needed because the rows keep attached and detached actors apart, and a canvas
/// draws both the same way — being drawn nowhere is a missing wire, not a
/// different sort of node.
impl Gathered {
    fn every_actor(&self) -> impl Iterator<Item = (Entity, &super::gather::ActorRow)> {
        self.rows
            .values()
            .flat_map(|row| row.actors.iter())
            .chain(self.detached.iter())
            .map(|actor| (actor.entity, actor))
    }
}

/// Kept for the `ParamKind` import the offer predicate needs.
const _: fn(&ParamKind) -> bool = |kind| kind.is_input();
