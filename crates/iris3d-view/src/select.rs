//! What the user currently has selected.
//!
//! A leaf module depending on nothing in the crate, because both halves of the
//! presentation layer need it and neither should depend on the other: the
//! interface reads and writes it, and the viewport reads it to draw the
//! selection outline and to decide what the transform handles move.
//!
//! One resource rather than a field on the interface's own state, so that
//! `viewport` does not have to reach into `ui` to find out what is highlighted.

use bevy::asset::AssetId;
use bevy::prelude::*;

use iris3d_scene::DataArray;

/// The selected object, actor, array and filter.
///
/// Four independent slots rather than one enum: the panel shows a different tab
/// for each, and switching tabs should not lose what was selected in the last
/// one. Selecting an actor does move [`object`](Self::object) to that actor's
/// source, so the outline, the tree highlight and the actor group agree.
#[derive(Resource, Default)]
pub struct Selection {
    /// The selected object.
    ///
    /// Read by `viewport::overlays` to draw the outline and by
    /// `viewport::manipulate` to decide what the handles move, as well as by
    /// both trees in the interface.
    pub object: Option<Entity>,
    pub actor: Option<Entity>,
    pub array: Option<AssetId<DataArray>>,
    /// By handle rather than entity, because a filter is named by handle
    /// everywhere a command speaks about one.
    pub filter: Option<u64>,
}
