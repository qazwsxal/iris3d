//! Launch options.
//!
//! Everything decided before the window opens and not changed afterwards. That
//! is a short list on purpose: the scene is driven over gRPC, so anything a
//! script can set belongs in the wire contract rather than here. What is left
//! is what a script *cannot* set — where to listen, and how to capture what was
//! drawn.
//!
//! There used to be a `--backend` here. One pathway is built now, so there is
//! nothing to choose between; a client asks `ListActorKinds` to learn what the
//! running one can draw, which was always the right question.
//!
//! Parsed before `App::new()`. Winit does not read `argv`, so nothing downstream
//! is surprised by the flags, and `--help` exits before a window is created.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "iris3d", about = "Scriptable 3D visualisation.")]
pub struct Cli {
    /// Address the gRPC control surface listens on.
    ///
    /// Loopback by default. iris3d is a desktop application scripted by
    /// processes on the same machine, so binding a wider interface should be a
    /// deliberate choice.
    #[arg(long, default_value = "[::1]:50051")]
    pub listen: SocketAddr,

    /// Save a screenshot to this path and quit.
    ///
    /// For checking what the backend actually drew. "It logged no errors" is a
    /// weaker claim than a picture, and a change to how something composites is
    /// only judgeable as an image.
    #[arg(long, value_name = "PATH")]
    pub screenshot: Option<PathBuf>,

    /// How many frames to render before the screenshot.
    ///
    /// Counted in frames because frames are what is being waited for: arrays
    /// arrive over gRPC, and the moment buffer needs the frame it was written
    /// in to have resolved.
    #[arg(long, default_value_t = 240, value_name = "N")]
    pub screenshot_after: u32,
}
