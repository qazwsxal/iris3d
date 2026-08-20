//! Whether the chosen pathway can run on this machine.
//!
//! Backends are picked at launch and never mixed, so "can this machine do it"
//! has to be answered once, before anything draws. A pathway that cannot run
//! **refuses**. It does not quietly fall back to another: the pathways
//! composite differently, so a substitution shows a picture that is wrong and
//! says nothing about it. Wrong and silent is worse than not starting.
//!
//! The check runs in [`Plugin::finish`] rather than
//! before `App::new()`. Requesting an adapter independently would test *an*
//! adapter rather than *the* adapter — Bevy picks by its own power preference
//! and backend order, and on a laptop with two GPUs that is a real difference.
//! By `finish` the render app exists and holds the adapter that will actually
//! draw, and the window has not opened yet, so a refusal still happens before
//! anything is on screen.

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::renderer::RenderAdapter;

use super::{BACKEND, REQUIRES};

/// Stops the app if the adapter is missing anything the pathway needs.
///
/// Silent when there is no render app at all, which is the headless case in
/// tests: there is no adapter to judge and nothing will be drawn either way.
pub(crate) fn refuse_unsupported(app: &App) {
    if REQUIRES.is_empty() {
        return;
    }

    let Some(render_app) = app.get_sub_app(RenderApp) else {
        return;
    };
    let Some(adapter) = render_app.world().get_resource::<RenderAdapter>() else {
        return;
    };

    let missing = REQUIRES - adapter.features();
    if missing.is_empty() {
        return;
    }

    // Naming what is missing is what says whether another machine would work.
    error!(
        "draw: the {BACKEND} backend needs GPU features this adapter does not have: {missing:?}"
    );
    error!(
        "draw: no fallback — there is no second pathway to substitute, and one that \
         composited differently would draw a picture that is wrong without saying so."
    );
    std::process::exit(1);
}
