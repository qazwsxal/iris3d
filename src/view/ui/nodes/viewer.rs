//! Drawing one node, and what clicking it does.
//!
//! [`Viewer`] is egui-snarl's per-node callback surface: titles, pins, bodies,
//! and what a dragged wire means when it lands. It holds a queue rather than the
//! world, so everything it can do to the scene is an action pushed onto that
//! queue — the same constraint the panel works under.

use super::*;

pub(super) struct Viewer<'a> {
    pub(super) scene: &'a Gathered,
    pub(super) actions: &'a mut PendingActions,
    pub(super) selection: &'a Selection,
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
                Some(row) => row.label.to_string(),
                None => format!("filter {id} — gone"),
            },
            Node::Actor(entity) => match self.scene.actor(*entity) {
                Some((_, row)) => row.label.to_string(),
                None => "gone".into(),
            },
            Node::Object(entity) => match self.scene.rows.get(entity) {
                Some(row) => row.name.clone(),
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
