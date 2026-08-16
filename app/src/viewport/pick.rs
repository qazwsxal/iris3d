//! Clicking something in the 3D view selects it.
//!
//! The panel's trees could select and the viewport could not, so switching
//! between them lost your place — and the most direct way of saying "that one"
//! was the one way the interface did not accept.
//!
//! # A real backend, not a hit test for the gizmo
//!
//! Nothing in the app raycast before this. It would have been less work to give
//! the transform gizmo its own analytic test against its three handles and stop
//! there, and it would have had to be thrown away: picking is what the
//! bidirectional event stream reports, and what `pick` and `hover` source nodes
//! will read. Building the general thing first means the gizmo sits on it rather
//! than beside it.
//!
//! # What is pickable
//!
//! **Placements**, which is where the meshes are. An actor owns the mesh its
//! kind builds but is permanently `Visibility::Hidden` — see
//! [`spawn_actor`](crate::scene::link::spawn_actor) — and a placement is a child
//! of the object carrying a clone of that handle. So a hit resolves to *this
//! drawing under this object*, which is exactly the pair the panel needs: an
//! actor drawn under three objects is three placements, and which one was
//! clicked is the only thing that says which object to highlight.
//!
//! `require_markers` is on, so nothing is pickable by accident. The gizmo's own
//! handles will want to be pickable without every mesh in the scene competing
//! with them, and a default of "everything with a mesh" makes that impossible to
//! arrange later.
//!
//! # What a pick does *not* carry yet
//!
//! The element index. Bevy's mesh backend computes a `triangle_index` while
//! raycasting and then drops it — `HitData::new` takes camera, depth, position
//! and normal, and leaves `extra` as `None`. So "which atom" needs its own
//! [`MeshRayCast`](bevy::picking::mesh_picking::ray_cast::MeshRayCast) call
//! rather than a read off the event, and that is the piece the event stream will
//! need. See `provenance` for the other half of the problem: with subsetting now
//! done by filters, a drawn index has to be walked back through the graph before
//! it means anything to a client.

use bevy::picking::mesh_picking::{MeshPickingCamera, MeshPickingPlugin, MeshPickingSettings};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;

use crate::scene::link::Placement;

/// Which placement was clicked, and the actor and object it resolves to.
///
/// A message rather than a direct write, because what *selection* means belongs
/// to the interface and picking should not depend on the interface existing.
/// `ui` reads these and turns them into the same `UiAction`s a tree click emits,
/// so there is still one path that changes what is selected.
#[derive(Message, Debug, Clone, Copy)]
pub struct Picked {
    /// The actor entity that was drawn.
    pub actor: Entity,
    /// The object it was drawn under. An actor appears once per object, so this
    /// is the one thing a hit can say that the actor alone cannot.
    pub object: Entity,
    /// Where the ray met the geometry, in world space.
    ///
    /// Free — the ray gave it up on the way to finding the entity — and it is
    /// what a client needs to place a label or measure a distance without
    /// knowing anything about what it hit. `None` if the backend did not report
    /// one.
    pub position: Option<Vec3>,
}

pub struct PickPlugin;

impl Plugin for PickPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MeshPickingPlugin)
            .insert_resource(MeshPickingSettings {
                // Opt in, so a mesh is pickable because something said so.
                require_markers: true,
                ..default()
            })
            .add_message::<Picked>()
            .add_systems(Update, (mark_camera, mark_placements))
            .add_observer(on_click);
    }
}

/// Lets the 3D camera cast picking rays.
///
/// The egui camera must not: it is a 2D camera on its own render layer, and
/// giving it a picking ray would put a second pointer into the same scene.
fn mark_camera(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<MeshPickingCamera>)>,
) {
    for camera in &cameras {
        commands.entity(camera).insert(MeshPickingCamera);
    }
}

/// Marks every placement pickable as it appears.
///
/// Placements are spawned and despawned as objects come and go, so this runs
/// every frame over whatever is new rather than being done once at startup.
fn mark_placements(
    mut commands: Commands,
    placements: Query<Entity, (With<Placement>, Without<Pickable>)>,
) {
    for placement in &placements {
        commands.entity(placement).insert(Pickable::default());
    }
}

/// Turns a click on a placement into a [`Picked`].
///
/// Left button only. The right and middle buttons pan the camera
/// ([`orbit_controls`](super::orbit_controls)), and a pan that ends on geometry
/// would otherwise also change the selection.
fn on_click(
    click: On<Pointer<Click>>,
    placements: Query<(&Placement, &ChildOf)>,
    mut picked: MessageWriter<Picked>,
) {
    if click.event.button != PointerButton::Primary {
        return;
    }
    // The hit target may be the placement itself or something a backend put
    // under it, so the placement is looked for rather than assumed.
    let Ok((placement, parent)) = placements.get(click.entity) else {
        return;
    };
    picked.write(Picked {
        actor: placement.0,
        object: parent.parent(),
        position: click.event.hit.position,
    });
}
