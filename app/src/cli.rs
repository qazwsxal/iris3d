//! Launch options.
//!
//! Everything decided before the window opens and not changed afterwards. That
//! is a short list on purpose: the scene is driven over gRPC, so anything a
//! script can set belongs in the wire contract rather than here. What is left
//! is what a script *cannot* set — where to listen, and which rendering pathway
//! to be.
//!
//! Parsed before `App::new()`. Winit does not read `argv`, so nothing downstream
//! is surprised by the flags, and `--help` exits before a window is created.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

use crate::draw::Backend;

#[derive(Parser, Debug)]
#[command(name = "iris3d", about = "Scriptable 3D visualisation.")]
pub struct Cli {
    /// Rendering pathway to run.
    ///
    /// A backend is a whole pipeline together with the actors built for it.
    /// They are mutually exclusive: two techniques that composite differently
    /// cannot share a frame correctly, so this is decided once, here, and not
    /// per camera or per object. Which actor kinds exist depends on it — ask
    /// `ListActorKinds` rather than assuming.
    #[arg(long, value_enum, default_value_t = Backend::Default)]
    pub backend: Backend,

    /// Address the gRPC control surface listens on.
    ///
    /// Loopback by default. iris3d is a desktop application scripted by
    /// processes on the same machine, so binding a wider interface should be a
    /// deliberate choice.
    #[arg(long, default_value = "[::1]:50051")]
    pub listen: SocketAddr,

    /// Save a screenshot to this path and quit.
    ///
    /// For checking what a backend actually drew. "It logged no errors" is a
    /// weaker claim than a picture, and two backends drawing the same scene are
    /// only comparable as images.
    #[arg(long, value_name = "PATH")]
    pub screenshot: Option<PathBuf>,

    /// How many frames to render before the screenshot.
    ///
    /// Counted in frames because frames are what is being waited for: arrays
    /// arrive over gRPC, acceleration structures build, and a raytraced image
    /// accumulates over several frames before it settles.
    #[arg(long, default_value_t = 240, value_name = "N")]
    pub screenshot_after: u32,
}
