//! What the user sees, and what they can do to it.
//!
//! Two halves that both act on the same selection:
//!
//! - [`viewport`] — the 3D view itself: the camera, picking, the transform
//!   handles and the read-only overlays.
//! - [`ui`] — the egui interface: a tabbed panel and a node canvas.
//! - [`select`] — what is currently selected, which both halves read and only
//!   the interface writes.
//!
//! These are one crate rather than two because the split between them is not a
//! layering: the interface reads what the viewport picked and the viewport draws
//! an outline around what the interface selected. Making them separate crates
//! would mean the selection resource lived below both in a crate of its own,
//! which buys nothing — nothing else would ever depend on it.
//!
//! Nothing below here knows this exists. A build with no interface still draws.

pub mod select;
pub mod ui;
pub mod viewport;

pub use ui::UiPlugin;
pub use viewport::ViewportPlugin;
