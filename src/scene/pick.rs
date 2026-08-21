//! What a click in the 3D view resolved to.
//!
//! The event lives here rather than with the viewport that raises it, because
//! three unrelated things read it — a `pick` source node in
//! `filter::source`, the watch stream in
//! `grpc::watch`, and the interface — and none of them
//! should depend on the presentation layer to do so. `viewport::pick` writes it;
//! everything else reads it.

use bevy::prelude::*;

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
    /// Which vertex of the drawn mesh was hit, if it could be recovered.
    ///
    /// Bevy's own event does not carry this — see the module doc — so it
    /// costs a second, entity-filtered raycast from the camera through
    /// [`position`](Self::position). `None` when the first cast reported no
    /// position, or the second could not resolve a triangle.
    ///
    /// This is a **drawn** index, into whatever mesh the actor's `geometry`
    /// input is bound to. Recovering the element a client uploaded means
    /// walking it back through `provenance`.
    pub element: Option<u32>,
}
