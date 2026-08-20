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

use crate::scene::DataArray;
use crate::select::Selection;
use iris3d_model::data as bound_handle;
use iris3d_model::{ParamKind, ParamValue};

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

/// How far from the left each node sits, by how far it is *along* the graph.
///
/// A column per kind was the first attempt and could not order a filter that
/// reads another filter's output: both are filters, so both landed in the same
/// column and the wire between them ran backwards. Depth is the property that
/// actually matters, and it is not a property of the kind.
///
/// The layer is the **longest** path from a source, not the shortest. A node is
/// placed after everything it reads, so an input arriving from two chains of
/// different lengths still lands to the left of it. Relaxed to a fixpoint rather
/// than sorted topologically — the same answer, and it does not care if the
/// graph is disconnected or, despite the backend's cycle check, somehow not a
/// DAG: the iteration cap ends it either way.
fn layers(scene: &Gathered) -> HashMap<Node, f32> {
    let mut edges: Vec<(Node, Node)> = Vec::new();
    for filter in &scene.filters {
        let node = Node::Filter(filter.id);
        for spec in inputs_of(filter.specs) {
            if let Some(handle) = bound_handle(&filter.params, spec.id) {
                edges.push((Node::Data(handle), node));
            }
        }
        for (_, handle) in &filter.outputs {
            edges.push((node, Node::Data(*handle)));
        }
    }
    for (entity, actor) in scene.every_actor() {
        for spec in inputs_of(actor.specs) {
            if let Some(handle) = bound_handle(&actor.params, spec.id) {
                edges.push((Node::Data(handle), Node::Actor(entity)));
            }
        }
    }
    for row in scene.rows.values() {
        for actor in &row.actors {
            edges.push((Node::Actor(actor.entity), Node::Object(row.entity)));
        }
        for child in &row.children {
            edges.push((Node::Object(*child), Node::Object(row.entity)));
        }
    }

    let mut depth: HashMap<Node, f32> = HashMap::new();
    for (from, to) in &edges {
        depth.entry(*from).or_insert(0.0);
        depth.entry(*to).or_insert(0.0);
    }
    // One pass per node is enough to settle the longest path; the cap is what
    // makes an unexpected cycle terminate instead of spinning.
    for _ in 0..depth.len().min(64) {
        let mut moved = false;
        for (from, to) in &edges {
            let after = depth.get(from).copied().unwrap_or(0.0) + 1.0;
            let slot = depth.entry(*to).or_insert(0.0);
            if after > *slot {
                *slot = after;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    depth
}

/// Where a node sits, given the depths worked out for the whole graph.
///
/// A node in no edge at all — an upload nothing reads, an object with nothing
/// under it — has no depth and starts at the left, which is where an unused
/// thing belongs.
fn column(node: &Node, depth: &HashMap<Node, f32>) -> f32 {
    40.0 + COLUMN * depth.get(node).copied().unwrap_or(0.0)
}

/// Roughly how tall a node will be, in pin rows.
///
/// Guessed rather than measured, because the layout runs before the node has
/// ever been drawn and egui only knows a widget's size afterwards. Guessing
/// high is the safe direction — too much space between nodes is untidy, too
/// little stacks them on top of each other.
/// Counts the body's controls as well as the pins, because they are what a node
/// is now mostly made of — a `volume` has three inputs and eight settings, and
/// leaving room for three stacked it on top of whatever was placed below it.
fn rows(node: &Node, scene: &Gathered) -> usize {
    let pins = match node {
        Node::Data(_) => 1,
        Node::Filter(id) => scene
            .filter(*id)
            .map_or(1, |row| inputs_of(row.specs).count().max(row.outputs.len())),
        Node::Actor(entity) => scene
            .actor(*entity)
            .map_or(1, |(_, row)| inputs_of(row.specs).count().max(1)),
        Node::Object(_) => 1,
    };
    let settings = settings_of(scene, node).map_or(0, |specs| {
        specs.iter().filter(|spec| is_setting(spec)).count()
    });
    // Bodies open by default, so the estimate assumes the whole set is showing,
    // plus the header row the busy kinds carry. Folding one afterwards leaves a
    // gap, which is the harmless direction to be wrong in — guessing low stacks
    // nodes on top of each other.
    pins + settings + usize::from(settings > INLINE_SETTINGS)
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
        wanted.extend(scene.detached.iter().map(|actor| Node::Actor(actor.entity)));

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
        //
        // Keyed on where a node actually *is* rather than where it would be put
        // now: depths shift as the graph grows, and a node the user has dragged
        // has no column at all. What matters is only that a new node does not
        // land on something already occupying that strip.
        let depth = layers(scene);
        let mut next_row: HashMap<u32, f32> = HashMap::new();
        for (pos, node) in self.snarl.nodes_pos() {
            let slot = next_row.entry(pos.x as u32).or_insert(40.0);
            *slot = slot.max(pos.y + 72.0 + 26.0 * rows(node, scene) as f32);
        }
        for node in wanted {
            if self.placed.contains_key(&node) {
                continue;
            }
            let x = column(&node, &depth);
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
                        input: PLACED_HERE,
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
                        input: NESTED_HERE,
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
    specs: &'static [iris3d_model::ParamSpec],
) -> impl Iterator<Item = &'static iris3d_model::ParamSpec> {
    specs.iter().filter(|spec| spec.kind.is_input())
}

/// The complement of [`inputs_of`]: everything that is a control, not a pin.
///
/// Defined against `is_input` rather than by listing the control kinds, so a
/// seventh `ParamKind` lands in exactly one of the two and cannot go missing
/// from both.
fn is_setting(spec: &iris3d_model::ParamSpec) -> bool {
    !spec.kind.is_input()
}

/// What a node declares, if it declares anything.
///
/// Objects and data handles have no kind behind them — an object is a place and
/// a handle is a name — so they have nothing to draw and say so here once.
fn settings_of(scene: &Gathered, node: &Node) -> Option<&'static [iris3d_model::ParamSpec]> {
    match node {
        Node::Filter(id) => scene.filter(*id).map(|row| row.specs),
        Node::Actor(entity) => scene.actor(*entity).map(|(_, row)| row.specs),
        Node::Data(_) | Node::Object(_) => None,
    }
}

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

struct Viewer<'a> {
    scene: &'a Gathered,
    actions: &'a mut PendingActions,
    selection: &'a Selection,
}

impl Viewer<'_> {
    /// Whether this node is what the rest of the interface is pointed at.
    fn is_selected(&self, node: &Node) -> bool {
        match node {
            Node::Object(entity) => self.selection.object == Some(*entity),
            Node::Actor(entity) => self.selection.actor == Some(*entity),
            Node::Filter(id) => self.selection.filter == Some(*id),
            Node::Data(handle) => {
                self.array_of(*handle).is_some() && self.selection.array == self.array_of(*handle)
            }
        }
    }

    /// The array a handle names, if it names one rather than a mesh.
    fn array_of(&self, handle: u64) -> Option<AssetId<DataArray>> {
        self.scene
            .held
            .iter()
            .find(|(_, (id, _))| *id == handle)
            .map(|(asset, _)| *asset)
    }

    /// The controls themselves, drawn from the same declaration the panel uses.
    ///
    /// Separate from `show_body` only because it is called from two places —
    /// folded and unfolded — and the difference between them should be the
    /// frame around the controls, never the controls.
    fn body_controls(&mut self, ui: &mut egui::Ui, node: Node) {
        let edits = match node {
            Node::Filter(id) => {
                let Some(row) = self.scene.filter(id) else {
                    return;
                };
                params::controls_where(
                    ui,
                    self.scene,
                    row.specs,
                    &row.params,
                    ("node-filter", id),
                    |_| None,
                    is_setting,
                )
            }
            Node::Actor(entity) => {
                let Some((_, row)) = self.scene.actor(entity) else {
                    return;
                };
                params::controls_where(
                    ui,
                    self.scene,
                    row.specs,
                    &row.params,
                    ("node-actor", entity),
                    |_| None,
                    is_setting,
                )
            }
            Node::Data(_) | Node::Object(_) => return,
        };
        for (param, value) in edits.set {
            match node {
                Node::Filter(id) => self
                    .actions
                    .0
                    .push(UiAction::SetFilterParam(id, param, value)),
                Node::Actor(entity) => self
                    .actions
                    .0
                    .push(UiAction::SetParam(entity, param, value)),
                Node::Data(_) | Node::Object(_) => {}
            }
        }
    }
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
            // Two, not one. Being drawn under an object and being nested inside
            // it are different relations — one is a placement, the other a
            // transform parent — and sharing a pin made them share a colour,
            // which is exactly the distinction the canvas is supposed to draw.
            Node::Object(_) => 2,
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
            Some(Node::Object(_)) if pin.id.input == PLACED_HERE => {
                ui.weak("drawn here");
                PinInfo::square().with_fill(PLACEMENT)
            }
            Some(Node::Object(_)) => {
                ui.weak("nested here");
                PinInfo::square().with_fill(NESTING)
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

    /// The title, and clicking it selects the thing the node stands for.
    ///
    /// The canvas selects as the panel's trees do, so switching views keeps
    /// where you were. Selection is already shared state — `viewport::overlays`
    /// draws the outline from it, both trees highlight from it — so the canvas
    /// joining in costs one action per kind and nothing else.
    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<Node>,
    ) {
        let Some(node) = snarl.get_node(node).copied() else {
            return;
        };
        let title = self.title(&node);
        let chosen = self.is_selected(&node);
        // A filter that ran and refused says so on the canvas, not only in the
        // panel — the canvas is where a chain is read, and "which link went
        // dead" is the question being asked while looking at it. The marker is
        // in the title so it survives a folded body.
        let problem = match node {
            Node::Filter(id) => self.scene.filter(id).and_then(|row| row.problem.clone()),
            _ => None,
        };
        let title = match &problem {
            Some(_) => egui::RichText::new(format!("⚠ {title}")).color(ui.visuals().error_fg_color),
            None => egui::RichText::new(title),
        };
        let mut clicked = ui.selectable_label(chosen, title);
        // Only when there is one: an empty tooltip on every healthy node is a
        // grey box that follows the pointer round the canvas.
        if let Some(problem) = &problem {
            clicked = clicked.on_hover_text(problem);
        }
        if clicked.clicked() {
            match node {
                Node::Object(entity) => self.actions.0.push(UiAction::Select(entity)),
                // Which object it was clicked under: an actor drawn under
                // several appears once here, so there is no row to answer it
                // and the panel's own resolution is what fills it in.
                Node::Actor(entity) => {
                    let under = self
                        .scene
                        .rows
                        .values()
                        .find(|row| row.actors.iter().any(|actor| actor.entity == entity))
                        .map(|row| row.entity);
                    self.actions.0.push(UiAction::SelectActor(entity, under));
                }
                Node::Filter(id) => self.actions.0.push(UiAction::SelectFilter(id)),
                // A handle names an array or a mesh, and only an array is
                // selectable in the Data tab. A mesh handle selects nothing
                // rather than selecting something else.
                Node::Data(handle) => {
                    if let Some(asset) = self.array_of(handle) {
                        self.actions.0.push(UiAction::SelectArray(asset));
                    }
                }
            }
        }
    }

    /// Whether the node has settings to show between its pins.
    ///
    /// Inputs are pins and everything else is a control, so a kind with nothing
    /// but inputs — `geometry`, and every data node — has no body at all and
    /// keeps the compact shape it had before bodies existed.
    fn has_body(&mut self, node: &Node) -> bool {
        settings_of(self.scene, node).is_some_and(|specs| specs.iter().any(is_setting))
    }

    /// The node's own controls, drawn from the same declaration the panel uses.
    ///
    /// This is what makes a maths node legible: a `compare` whose threshold is
    /// only visible in another view is not a node, it is a node-shaped label.
    /// `ui::params::controls_where` is the panel's own function, so a new kind
    /// or a new parameter arrives here with no work, and the two views cannot
    /// disagree about what a control does.
    ///
    /// Offers are declined. An offer belongs beside an empty input picker, and
    /// inputs are pins here — the equivalent gesture is dragging a wire.
    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<Node>,
    ) {
        let Some(node) = snarl.get_node(node).copied() else {
            return;
        };
        // **Vertical, explicitly.** The body is drawn inside snarl's own
        // horizontal run — pins left, body, pins right — so a control inherits
        // that direction and every one of them lands on the same line, drawn
        // over the last. `set_max_width` does not save it: the widgets do not
        // wrap, they overlap.
        let count = settings_of(self.scene, &node).map_or(0, |specs| {
            specs.iter().filter(|spec| is_setting(spec)).count()
        });
        ui.vertical(|ui| {
            // A slider fills what it is given, so it has to be told. Wider than
            // this and one filter's controls set the width of the whole column.
            ui.set_max_width(BODY_WIDTH);
            ui.spacing_mut().slider_width = 90.0;
            // Everything is open until someone shuts it. A busy kind — nine
            // controls on `cartoon`, eight on `volume` — gets a header so it
            // *can* be folded away when the wires behind it matter more, but
            // hiding a control the user never asked to hide is the worse
            // default: stage 3's whole point is that these are draggable, and a
            // slider behind a click is not.
            if count > INLINE_SETTINGS {
                let open = egui::CollapsingHeader::new(format!("{count} settings"))
                    .id_salt(("node-body", node))
                    .default_open(true);
                open.show(ui, |ui| self.body_controls(ui, node));
            } else {
                self.body_controls(ui, node);
            }
        });
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
        // Placement and nesting, which join the object pin rather than a
        // parameter. Both replace a whole set, so they are read out of the scene
        // and sent back with one more member.
        if let Node::Object(object) = sink {
            let Some(row) = self.scene.rows.get(&object) else {
                return;
            };
            // Which pin it lands on decides what it means, so a wire cannot be
            // dropped on "drawn here" and quietly become a nesting.
            match (to.id.input, source) {
                (PLACED_HERE, Node::Actor(entity)) => {
                    let Some((_, actor)) = self.scene.actor(entity) else {
                        return;
                    };
                    let mut parents = self.scene.objects_of(entity);
                    if !parents.contains(&row.id) {
                        parents.push(row.id);
                    }
                    self.actions
                        .0
                        .push(UiAction::SetActorParents(actor.id, parents));
                }
                (NESTED_HERE, Node::Object(child)) => {
                    // An object cannot be nested in itself, and the backend
                    // refuses a cycle beyond that with `WouldCycle`.
                    if child != object
                        && let Some(child) = self.scene.rows.get(&child)
                    {
                        self.actions
                            .0
                            .push(UiAction::SetObjectParent(child.id, Some(row.id)));
                    }
                }
                _ => {}
            }
            return;
        }

        // Everything else is data, and only a data handle can feed a parameter.
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

    /// Likewise: cutting a wire is a scene change, not a canvas one.
    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<Node>) {
        let (Some(source), Some(sink)) = (
            snarl.get_node(from.id.node).copied(),
            snarl.get_node(to.id.node).copied(),
        ) else {
            return;
        };
        // Taking an actor off an object, or an object out of its parent. The
        // mirror of `connect`: the set is read out and sent back one shorter.
        if let Node::Object(object) = sink {
            let Some(row) = self.scene.rows.get(&object) else {
                return;
            };
            match (to.id.input, source) {
                (PLACED_HERE, Node::Actor(entity)) => {
                    let Some((_, actor)) = self.scene.actor(entity) else {
                        return;
                    };
                    let mut parents = self.scene.objects_of(entity);
                    parents.retain(|id| *id != row.id);
                    self.actions
                        .0
                        .push(UiAction::SetActorParents(actor.id, parents));
                }
                (NESTED_HERE, Node::Object(child)) => {
                    if let Some(child) = self.scene.rows.get(&child) {
                        self.actions
                            .0
                            .push(UiAction::SetObjectParent(child.id, None));
                    }
                }
                _ => {}
            }
            return;
        }

        // Unbinding a parameter. Only worth asking for where the input is
        // optional — clearing a required one leaves something that cannot draw,
        // which the backend refuses anyway, so refusing here keeps the wire on
        // screen rather than having it vanish and come back.
        let spec = match sink {
            Node::Filter(id) => self
                .scene
                .filter(id)
                .and_then(|row| inputs_of(row.specs).nth(to.id.input)),
            Node::Actor(entity) => self
                .scene
                .actor(entity)
                .and_then(|(_, row)| inputs_of(row.specs).nth(to.id.input)),
            Node::Data(_) | Node::Object(_) => None,
        };
        let Some(spec) = spec.filter(|spec| !spec.kind.is_required()) else {
            return;
        };
        match sink {
            Node::Filter(id) => {
                self.actions
                    .0
                    .push(UiAction::SetFilterParam(id, spec.id, ParamValue::Unset))
            }
            Node::Actor(entity) => {
                self.actions
                    .0
                    .push(UiAction::SetParam(entity, spec.id, ParamValue::Unset))
            }
            Node::Data(_) | Node::Object(_) => {}
        }
    }

    /// Right-clicking a pin drops everything on it, one wire at a time.
    fn drop_inputs(&mut self, pin: &InPin, snarl: &mut Snarl<Node>) {
        for from in pin.remotes.clone() {
            let from = OutPin {
                id: from,
                remotes: Vec::new(),
            };
            self.disconnect(&from, pin, snarl);
        }
    }

    fn drop_outputs(&mut self, pin: &OutPin, snarl: &mut Snarl<Node>) {
        for to in pin.remotes.clone() {
            let to = InPin {
                id: to,
                remotes: Vec::new(),
            };
            self.disconnect(pin, &to, snarl);
        }
    }
}

/// An input row's label, and whether anything is bound to it.
///
/// A required input with no wire is the one thing worth shouting about: it is
/// why a filter produces nothing and why an actor draws nothing, and on a canvas
/// the absence of a wire is easy to miss among the ones that are there.
fn label_for(ui: &mut egui::Ui, spec: Option<&'static iris3d_model::ParamSpec>, pin: &InPin) {
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
