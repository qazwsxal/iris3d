//! Where a node sits on the canvas, and which wires exist.
//!
//! The canvas is a projection of the scene, rebuilt every frame from it rather
//! than being state of its own — so a change made in the panel appears here and
//! a change made here goes out as the same command the panel sends.
//!
//! [`layers`] is the interesting part: depth is what decides the column, not
//! what kind of node it is. Two filters where one reads the other belong in
//! different columns, and that is not a property of either kind.

use super::*;

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
pub(super) fn layers(scene: &Gathered) -> HashMap<Node, f32> {
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
pub(super) fn column(node: &Node, depth: &HashMap<Node, f32>) -> f32 {
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
pub(super) fn rows(node: &Node, scene: &Gathered) -> usize {
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
    pub(super) fn rewire(&mut self, scene: &Gathered) {
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

    pub(super) fn wire_data(&mut self, handle: u64, to: NodeId, input: usize) {
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
pub(super) fn inputs_of(
    specs: &'static [crate::model::ParamSpec],
) -> impl Iterator<Item = &'static crate::model::ParamSpec> {
    specs.iter().filter(|spec| spec.kind.is_input())
}

/// The complement of [`inputs_of`]: everything that is a control, not a pin.
///
/// Defined against `is_input` rather than by listing the control kinds, so a
/// seventh `ParamKind` lands in exactly one of the two and cannot go missing
/// from both.
pub(super) fn is_setting(spec: &crate::model::ParamSpec) -> bool {
    !spec.kind.is_input()
}

/// What a node declares, if it declares anything.
///
/// Objects and data handles have no kind behind them — an object is a place and
/// a handle is a name — so they have nothing to draw and say so here once.
pub(super) fn settings_of(
    scene: &Gathered,
    node: &Node,
) -> Option<&'static [crate::model::ParamSpec]> {
    match node {
        Node::Filter(id) => scene.filter(*id).map(|row| row.specs),
        Node::Actor(entity) => scene.actor(*entity).map(|(_, row)| row.specs),
        Node::Data(_) | Node::Object(_) => None,
    }
}
