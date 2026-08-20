//! When the app draws.
//!
//! Nothing in iris3d moves on its own. The picture changes when a client sends
//! a command, or when someone moves the camera or touches the UI, and it is
//! identical between those moments. Drawing at the refresh rate anyway would
//! keep a GPU busy re-rendering the same frame — which costs battery on a
//! laptop, and costs more than that when the same machine is computing the data
//! being looked at.
//!
//! So the window is reactive: winit sleeps until an event arrives. That takes
//! two supports, and this module owns both. An idle tick, so a missed event
//! cannot leave a stale window forever. And [`KeepAwake`], for work that does
//! not finish in the frame that started it.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::window::RequestRedraw;
use bevy::winit::{UpdateMode, WinitSettings};
use std::time::Duration;

/// Longest the loop may sleep while the window has focus.
///
/// A backstop only. Real changes arrive as events, so this is about recovering
/// from a missed one, not about keeping up with anything.
const IDLE_TICK: Duration = Duration::from_secs(5);

/// The same backstop with the window in the background, where nobody is waiting
/// on the picture. A gRPC command still wakes the loop immediately, so a script
/// does not have to wait this out.
const BACKGROUND_IDLE_TICK: Duration = Duration::from_secs(60);

/// How long to keep drawing after the last sign of activity.
///
/// A command does not finish in the frame it is applied: entities are spawned
/// through deferred `Commands`, meshes and bounds appear a tick later, render
/// pipelines are specialised after that, and framing deliberately waits for the
/// scene to go quiet first. A second of frames covers all of it, and still
/// leaves the GPU idle the rest of the time.
const TAIL: Duration = Duration::from_secs(1);

pub struct RedrawPlugin;

impl Plugin for RedrawPlugin {
    fn build(&self, app: &mut App) {
        // Matches `WinitSettings::desktop_app()`, written out because the two
        // waits are a decision rather than a default: iris3d is a viewer, not a
        // game, and it has a wake-up path of its own for scripted changes.
        app.insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive(IDLE_TICK),
            unfocused_mode: UpdateMode::reactive_low_power(BACKGROUND_IDLE_TICK),
        })
        .init_resource::<KeepAwake>()
        // Coming up takes several frames: shaders, the egui context, the first
        // camera. None of that is driven by an event.
        .add_systems(Startup, |mut awake: ResMut<KeepAwake>| awake.nudge())
        // In `Last`, so it sees everything the frame did before deciding
        // whether another frame is needed.
        .add_systems(Last, keep_awake);
    }
}

/// Holds the update loop open a moment longer.
///
/// Anything that starts work it cannot finish in one frame calls
/// [`nudge`](Self::nudge). Without that the loop sleeps again as soon as the
/// event that woke it is handled, and the result of the work would not reach
/// the screen until something else happened to wake it.
#[derive(Resource, Default)]
pub struct KeepAwake(Option<Timer>);

impl KeepAwake {
    /// Asks for frames to keep coming for [`TAIL`] longer.
    pub fn nudge(&mut self) {
        self.0 = Some(Timer::new(TAIL, TimerMode::Once));
    }
}

/// Anything that has just appeared or just moved. The signal that the scene is
/// not yet at rest.
type Settling<'w, 's> = Query<'w, 's, (), Or<(Added<Aabb>, Changed<GlobalTransform>)>>;
/// Asks for the next frame for as long as the scene is still settling.
fn keep_awake(
    time: Res<Time>,
    mut awake: ResMut<KeepAwake>,
    settling: Settling,
    mut resized: MessageReader<bevy::window::WindowResized>,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    // The same signals framing waits on: new geometry, and anything moving.
    if !settling.is_empty() {
        awake.nudge();
    }

    // A resize changes every pixel without moving anything in the scene, so
    // none of the signals above notice it. Winit wakes the loop for the resize
    // itself, but one frame is not always enough: the render targets are
    // rebuilt at the new size, and a pathway that accumulates over frames has
    // just had its history invalidated. Without a tail here the previous
    // frame's interface stays on screen next to the new one.
    if !resized.is_empty() {
        resized.clear();
        awake.nudge();
    }

    let Some(timer) = awake.0.as_mut() else {
        return;
    };
    // A tail is always spent on back-to-back frames: every tick that has not
    // run out asks for the next one, so the loop does not get to sleep in the
    // middle of one. That matters because this is virtual time, which clamps a
    // long delta — a tail interrupted by a sleep would take far longer than
    // [`TAIL`] to run out.
    if timer.tick(time.delta()).is_finished() {
        awake.0 = None;
        return;
    }
    redraw.write(RequestRedraw);
}
