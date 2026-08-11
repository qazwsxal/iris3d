//! Screenshot on a timer, for looking at what actually rendered.
//!
//! A rendering backend can only really be checked by looking at it, and "it
//! logged no errors" is not the same claim. This captures the window to a file
//! after a fixed number of frames and then quits, which makes a render
//! inspectable without a person at the keyboard — and makes two backends
//! comparable by diffing their images.
//!
//! Frames rather than seconds because the thing being waited for is frames:
//! assets upload, acceleration structures build, and a raytraced image needs
//! several frames of accumulation before it settles.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

/// How long to keep running after the capture is requested, so the render
/// thread can read the target back and the file can be written before the app
/// tears down.
const GRACE_FRAMES: u32 = 60;

pub struct CapturePlugin {
    pub path: PathBuf,
    pub after: u32,
}

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Capture {
            path: self.path.clone(),
            after: self.after,
        })
        .add_systems(Update, capture);
    }
}

#[derive(Resource)]
struct Capture {
    path: PathBuf,
    after: u32,
}

fn capture(
    mut commands: Commands,
    capture: Res<Capture>,
    mut awake: ResMut<crate::redraw::KeepAwake>,
    mut quit: MessageWriter<AppExit>,
    mut frame: Local<u32>,
) {
    // The window is reactive: it sleeps until an event arrives, and an idle app
    // ticks every five seconds. A frame count would take a quarter of an hour
    // to reach its target that way, so hold the loop awake until the capture is
    // done and let it go back to sleep afterwards.
    awake.nudge();
    *frame += 1;
    if *frame == capture.after {
        info!("capture: taking a screenshot into {}", capture.path.display());
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.path.clone()));
    }
    if *frame >= capture.after + GRACE_FRAMES {
        quit.write(AppExit::Success);
    }
}
