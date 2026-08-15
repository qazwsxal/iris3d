//! The node view: the scene as a graph, wired together by hand.
//!
//! The tabbed panel lists each kind of thing separately and says in words what
//! is connected to what — "from [3] colour map" on an input, "→ [9] surface" on
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

use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32};
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer, SnarlWidget};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};

use crate::scene::registry::{ParamKind, ParamValue, data as bound_handle};

use super::gather::Gathered;
use super::{PendingActions, UiAction};

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
fn column(node: &Node, scene: &Gathered) -> f32 {
    match node {
        // An upload starts a chain; a filter's output continues one. Putting
        // both in the leftmost column drew every produced handle to the left of
        // the filter that writes it, so those wires ran backwards against the
        // direction everything else flows.
        Node::Data(handle) => match scene.producers.contains_key(handle) {
            false => 40.0,
            true => 880.0,
        },
        Node::Filter(_) => 460.0,
        Node::Actor(_) => 1300.0,
        Node::Object(_) => 1720.0,
    }
}

/// Roughly how tall a node will be, in pin rows.
///
/// Guessed rather than measured, because the layout runs before the node has
/// ever been drawn and egui only knows a widget's size afterwards. Guessing
/// high is the safe direction — too much space between nodes is untidy, too
/// little stacks them on top of each other.
fn rows(node: &Node, scene: &Gathered) -> usize {
    match node {
        Node::Data(_) => 1,
        Node::Filter(id) => scene.filter(*id).map_or(1, |row| {
            inputs_of(row.specs).count().max(row.outputs.len())
        }),
        Node::Actor(entity) => scene
            .actor(*entity)
            .map_or(1, |(_, row)| inputs_of(row.specs).count().max(1)),
        Node::Object(_) => 1,
    }
}

impl NodeGraph {
    /// Brings the canvas into line with the scene, keeping positions.
    pub fn reconcile(&mut self, scene: &Gathered) {
        let mut wanted: Vec<Node> = Vec::new();
        wanted.extend(scene.bindable.iter().map(|(id, _)| Node::Data(*id)));
        wanted.extend(scene.filters.iter().map(|row| Node::Filter(row.id)));
        wanted.extend(scene.ordered.iter().map(|entity| Node::Object(*entity)));
        for row in scene.rows.values() {
            wanted.extend(row.actors.iter().map(|actor| Node::Actor(actor.entity)));
        }
        wanted.extend(
            scene
                .detached
                .iter()
                .map(|actor| Node::Actor(actor.entity)),
        );

        // Gone from the scene, so gone from the canvas. Removing by node rather
        // than clearing keeps every surviving node's position.
        let live: std::collections::HashSet<Node> = wanted.iter().copied().collect();
        let stale: Vec<Node> = self
            .placed
            .keys()
            .copied()
            .filter(|node| !live.contains(node))
            .collect();
        for node in stale {
            if let Some(id) = self.placed.remove(&node) {
                self.snarl.remove_node(id);
            }
        }

        // Down each column in turn, leaving room for however many rows the node
        // will have. Starting below the top edge rather than at zero: the canvas
        // opens showing the origin, and a node at y = 0 sits half under the menu
        // bar with its title clipped away.
        //
        // Seeded from where nodes already are, which is not a nicety. A scene
        // arrives over many frames — a script uploads arrays, then builds a
        // filter, then an actor — so most reconciles add one node to a canvas
        // that already has some. Starting each column at the top every time
        // stacked every later arrival on top of the first, and the result read
        // as nodes *missing* rather than as nodes overlapping.
        let mut next_row: HashMap<u32, f32> = HashMap::new();
        for (pos, node) in self.snarl.nodes_pos() {
            let slot = next_row.entry(column(node, scene) as u32).or_insert(40.0);
            *slot = slot.max(pos.y + 72.0 + 26.0 * rows(node, scene) as f32);
        }
        for node in wanted {
            if self.placed.contains_key(&node) {
                continue;
            }
            let x = column(&node, scene);
            let top = next_row.entry(x as u32).or_insert(40.0);
            let id = self.snarl.insert_node(egui::pos2(x, *top), node);
            *top += 72.0 + 26.0 * rows(&node, scene) as f32;
            self.placed.insert(node, id);
        }
    }

    /// Rewrites every wire from the scene.
    ///
    /// Cheap enough to do wholesale — the wire list is small and rebuilding it
    /// is what keeps the canvas from ever disagreeing with the world. A binding
    /// the backend refused simply never appears, which is the honest outcome.
    fn rewire(&mut self, scene: &Gathered) {
        // No wholesale clear in the API, so drop what it reports.
        for (from, to) in self.snarl.wires().collect::<Vec<_>>() {
            self.snarl.disconnect(from, to);
        }

        // Data flow: a handle's producer to whoever reads it.
        for filter in &scene.filters {
            let Some(&to) = self.placed.get(&Node::Filter(filter.id)) else {
                continue;
            };
            for (index, spec) in inputs_of(filter.specs).enumerate() {
                let Some(handle) = bound_handle(&filter.params, spec.id) else {
                    continue;
                };
                self.wire_data(handle, to, index);
            }
            // And its outputs into the data nodes they fill.
            for (slot, (_, handle)) in filter.outputs.iter().enumerate() {
                let Some(&data) = self.placed.get(&Node::Data(*handle)) else {
                    continue;
                };
                let Some(&from) = self.placed.get(&Node::Filter(filter.id)) else {
                    continue;
                };
                self.snarl.connect(
                    egui_snarl::OutPinId {
                        node: from,
                        output: slot,
                    },
                    egui_snarl::InPinId {
                        node: data,
                        input: 0,
                    },
                );
            }
        }

        for (entity, actor) in scene.every_actor() {
            let Some(&to) = self.placed.get(&Node::Actor(entity)) else {
                continue;
            };
            for (index, spec) in inputs_of(actor.specs).enumerate() {
                let Some(handle) = bound_handle(&actor.params, spec.id) else {
                    continue;
                };
                self.wire_data(handle, to, index);
            }
        }

        // Placement: an actor into each object it is drawn under. Its pin is
        // after the data inputs, so its index depends on the kind.
        for row in scene.rows.values() {
            let Some(&object) = self.placed.get(&Node::Object(row.entity)) else {
                continue;
            };
            for actor in &row.actors {
                let Some(&from) = self.placed.get(&Node::Actor(actor.entity)) else {
                    continue;
                };
                self.snarl.connect(
                    egui_snarl::OutPinId {
                        node: from,
                        output: 0,
                    },
                    egui_snarl::InPinId {
                        node: object,
                        input: 0,
                    },
                );
            }
            // Nesting: a child object into its parent.
            for child in &row.children {
                let Some(&from) = self.placed.get(&Node::Object(*child)) else {
                    continue;
                };
                self.snarl.connect(
                    egui_snarl::OutPinId {
                        node: from,
                        output: 0,
                    },
                    egui_snarl::InPinId {
                        node: object,
                        input: 0,
                    },
                );
            }
        }
    }

    fn wire_data(&mut self, handle: u64, to: NodeId, input: usize) {
        let Some(&from) = self.placed.get(&Node::Data(handle)) else {
            return;
        };
        self.snarl.connect(
            egui_snarl::OutPinId {
                node: from,
                output: 0,
            },
            egui_snarl::InPinId { node: to, input },
        );
    }
}

/// A kind's inputs, in declaration order.
///
/// The pin index *is* the position in this sequence, so both the drawing and
/// the wiring have to walk it the same way — hence one function rather than two
/// filters that could drift apart.
fn inputs_of(
    specs: &'static [crate::scene::registry::ParamSpec],
) -> impl Iterator<Item = &'static crate::scene::registry::ParamSpec> {
    specs.iter().filter(|spec| spec.kind.is_input())
}

pub fn show(
    ui: &mut egui::Ui,
    graph: &mut NodeGraph,
    scene: &Gathered,
    actions: &mut PendingActions,
) {
    graph.reconcile(scene);
    graph.rewire(scene);

    let mut viewer = Viewer { scene, actions };
    SnarlWidget::new()
        .id(egui::Id::new("iris3d-nodes"))
        .style(SnarlStyle::new())
        .show(&mut graph.snarl, &mut viewer, ui);
}

struct Viewer<'a> {
    scene: &'a Gathered,
    actions: &'a mut PendingActions,
}

impl SnarlViewer<Node> for Viewer<'_> {
    fn title(&mut self, node: &Node) -> String {
        match node {
            Node::Data(handle) => self.scene.describe_handle(*handle),
            Node::Filter(id) => match self.scene.filter(*id) {
                Some(row) => format!("[{}] {}", row.id, row.label),
                None => format!("[{id}] gone"),
            },
            Node::Actor(entity) => match self.scene.actor(*entity) {
                Some((_, row)) => format!("[{}] {}", row.id, row.label),
                None => "gone".into(),
            },
            Node::Object(entity) => match self.scene.rows.get(entity) {
                Some(row) => format!("[{}] {}", row.id, row.name),
                None => "gone".into(),
            },
        }
    }

    fn inputs(&mut self, node: &Node) -> usize {
        match node {
            // An upload has nothing feeding it; a filter's output does, and the
            // one pin is where that wire lands.
            Node::Data(_) => 1,
            Node::Filter(id) => self
                .scene
                .filter(*id)
                .map_or(0, |row| inputs_of(row.specs).count()),
            Node::Actor(entity) => self
                .scene
                .actor(*entity)
                .map_or(0, |(_, row)| inputs_of(row.specs).count()),
            // Everything drawn under it, and every object nested in it, arrive
            // on one pin: they are the same relation from the object's side.
            Node::Object(_) => 1,
        }
    }

    fn outputs(&mut self, node: &Node) -> usize {
        match node {
            Node::Data(_) => 1,
            Node::Filter(id) => self.scene.filter(*id).map_or(0, |row| row.outputs.len()),
            // Where it is drawn.
            Node::Actor(_) => 1,
            // Its parent object.
            Node::Object(_) => 1,
        }
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<Node>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node = snarl.get_node(pin.id.node).copied();
        match node {
            Some(Node::Data(_)) => {
                ui.weak("from");
                PinInfo::circle().with_fill(DATA)
            }
            Some(Node::Filter(id)) => {
                let spec = self
                    .scene
                    .filter(id)
                    .and_then(|row| inputs_of(row.specs).nth(pin.id.input));
                label_for(ui, spec, pin);
                PinInfo::circle().with_fill(DATA)
            }
            Some(Node::Actor(entity)) => {
                let spec = self
                    .scene
                    .actor(entity)
                    .and_then(|(_, row)| inputs_of(row.specs).nth(pin.id.input));
                label_for(ui, spec, pin);
                PinInfo::circle().with_fill(DATA)
            }
            Some(Node::Object(_)) => {
                ui.weak("contains");
                PinInfo::square().with_fill(PLACEMENT)
            }
            None => PinInfo::circle(),
        }
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<Node>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let node = snarl.get_node(pin.id.node).copied();
        match node {
            Some(Node::Data(handle)) => {
                let described = self
                    .scene
                    .bindable
                    .iter()
                    .find(|(id, _)| *id == handle)
                    .map(|(_, meta)| meta.describe())
                    .unwrap_or_else(|| "gone".into());
                ui.weak(described);
                PinInfo::circle().with_fill(DATA)
            }
            Some(Node::Filter(id)) => {
                if let Some((spec, _)) = self
                    .scene
                    .filter(id)
                    .and_then(|row| row.outputs.get(pin.id.output))
                {
                    ui.label(spec.label);
                }
                PinInfo::circle().with_fill(DATA)
            }
            Some(Node::Actor(_)) => {
                ui.weak("drawn under");
                PinInfo::square().with_fill(PLACEMENT)
            }
            Some(Node::Object(_)) => {
                ui.weak("inside");
                PinInfo::square().with_fill(NESTING)
            }
            None => PinInfo::circle(),
        }
    }

    /// A wire the user dragged.
    ///
    /// Nothing is connected here. The default implementation would call
    /// `snarl.connect` and the canvas would show a wire the scene knows nothing
    /// about — which would then vanish on the next frame's rewire, looking like
    /// the drag failed. Emitting the action instead sends it down the same path
    /// the panel's pickers use, so it is validated once, in one place, and the
    /// wire appears because the binding exists.
    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<Node>) {
        let (Some(source), Some(sink)) = (
            snarl.get_node(from.id.node).copied(),
            snarl.get_node(to.id.node).copied(),
        ) else {
            return;
        };
        // Only a data handle can feed a parameter. Placement and nesting are not
        // editable here yet, so a wire between those pins is simply refused
        // rather than silently doing nothing else.
        let Node::Data(handle) = source else {
            return;
        };
        match sink {
            Node::Filter(id) => {
                if let Some(spec) = self
                    .scene
                    .filter(id)
                    .and_then(|row| inputs_of(row.specs).nth(to.id.input))
                {
                    self.actions.0.push(UiAction::SetFilterParam(
                        id,
                        spec.id,
                        ParamValue::Data(handle),
                    ));
                }
            }
            Node::Actor(entity) => {
                if let Some(spec) = self
                    .scene
                    .actor(entity)
                    .and_then(|(_, row)| inputs_of(row.specs).nth(to.id.input))
                {
                    self.actions.0.push(UiAction::SetParam(
                        entity,
                        spec.id,
                        ParamValue::Data(handle),
                    ));
                }
            }
            Node::Data(_) | Node::Object(_) => {}
        }
    }

    /// Likewise: dropping a wire is a scene change, not a canvas one.
    ///
    /// Unbinding is not expressible yet — every setter takes a value, and there
    /// is no `ParamValue` meaning "nothing". Refusing is honest until there is.
    fn disconnect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<Node>) {}

    fn drop_inputs(&mut self, _pin: &InPin, _snarl: &mut Snarl<Node>) {}

    fn drop_outputs(&mut self, _pin: &OutPin, _snarl: &mut Snarl<Node>) {}
}

/// An input row's label, and whether anything is bound to it.
///
/// A required input with no wire is the one thing worth shouting about: it is
/// why a filter produces nothing and why an actor draws nothing, and on a canvas
/// the absence of a wire is easy to miss among the ones that are there.
fn label_for(
    ui: &mut egui::Ui,
    spec: Option<&'static crate::scene::registry::ParamSpec>,
    pin: &InPin,
) {
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
