//! The plumbing every layer uses and none of them owns.
//!
//! Three unrelated things live here, and they are together because they share
//! one property: everything above needs them, and nothing here needs anything
//! else in iris3d. That is what makes this the bottom of the graph.
//!
//! - [`bus`] — the channel commands arrive on, generic over what they are.
//!   The scene and the filter graph each own one.
//! - [`counter`] — the handles clients name things by. One sequence for
//!   objects, actors, filters and arrays alike, so a handle is never ambiguous
//!   about what it refers to.
//! - [`redraw`] — when the app draws at all. iris3d is event-driven, not
//!   frame-driven, and that policy has to be visible to everything that changes
//!   something a viewer would see.
//!
//! Nothing here knows what an array is, what a scene is, or how anything is
//! drawn. If something added here would need to, it belongs a layer up.

pub mod bus;
pub mod counter;
pub mod redraw;

pub use bus::{Bus, BusSender, Gone};
pub use counter::{CounterPlugin, GlobalIDCounter, UniqueID};
pub use redraw::{KeepAwake, RedrawPlugin};
