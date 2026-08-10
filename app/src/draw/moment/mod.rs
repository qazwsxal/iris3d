//! Moment-based order-independent transparency for absorbing volumes.
//!
//! A second way of drawing a closed mesh, standing beside [`super::surface`]
//! rather than replacing it. Where `surface` draws a shell, this treats the mesh
//! as the boundary of a solid that absorbs light at a uniform rate. What you see
//! is the interior: thick parts read dark, thin parts read clear, and nested or
//! overlapping volumes compose correctly whatever order they are drawn in.
//!
//! # Why not ordinary alpha blending
//!
//! Alpha blending is multiplicative, so it depends on the order fragments
//! arrive in. Sorting triangles fixes it only for meshes that do not intersect
//! and are not nested — which is exactly the case scientific data does not obey.
//!
//! Absorbance is additive, and addition does not care about order:
//!
//! ```text
//! A(z) = -ln T(z) = sum of the absorbance of everything in front of z
//! ```
//!
//! So the depth-dependent absorbance is accumulated with an additive blend,
//! then transmittance `T = exp(-A)` is reconstructed in a second pass. No
//! sorting, no per-pixel lists, and a fixed cost per pixel.
//!
//! # The signed-prefix trick
//!
//! A fragment must contribute using **only its own depth** — that is what makes
//! the accumulation order-independent. A back face does not know which front
//! face it pairs with, and must not need to.
//!
//! With uniform extinction `sigma` the absorbance has a density rather than a
//! spike, so it has an antiderivative:
//!
//! ```text
//! dA/dz = sigma inside the mesh, 0 outside
//! F(z)  = sigma * z
//! ```
//!
//! An interior interval contributes `F(z_out) - F(z_in)`. Front faces therefore
//! add `-F(z)` and back faces add `+F(z)`, and the additive blend performs the
//! pairing on its own. Non-convex and nested meshes need no special handling,
//! which is the whole point.
//!
//! One draw does both signs: `cull_mode: None` and a branch on
//! `@builtin(front_facing)`.
//!
//! # Where this is in the build order
//!
//! This is `k = 0` only — a signed thickness pass, milestones 1 and 2 of the
//! reference document's build order. For a single convex volume `exp(-A)` is
//! not an approximation but the exact answer, which makes this both a working
//! renderer and the reference image that later moment counts are checked
//! against. Higher moments come next, and until they land a *concave* volume
//! renders its total thickness correctly but cannot be occluded partway through
//! by other transparent geometry.
//!
//! See `ref/mboit-bevy-reference.md`. Two of its notes do not survive contact
//! with Bevy 0.19 and are corrected here — see [`pass`] for the depth handling
//! and [`pipeline`] for the depth comparison.

use bevy::asset::embedded_asset;
use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::{SpecializedMeshPipelines, WgpuFeatures};
use bevy::render::renderer::RenderDevice;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

mod extract;
mod pass;
mod pipeline;
mod prepare;
mod validate;

use pass::{moment_pass, moment_resolve};
use pipeline::{init_moment_pipelines, queue_moment_pipelines};
use prepare::{prepare_moment_bind_groups, prepare_moment_instances, prepare_moment_textures};

/// Switches the moment passes on for one camera.
///
/// Per-camera rather than global because the passes cost an extra draw of every
/// volume, and a thumbnail or a picking view should not pay for them.
///
/// Carries no depth bound yet. At `k = 0` the accumulated quantity is an
/// ordinary optical depth in world units and needs no warping; the bound
/// arrives with the higher moments, which do — see the reference document's §4.
#[derive(Component, Debug, Clone, Copy, Default, ExtractComponent)]
pub struct MomentTransparency;

/// Marks a mesh as the boundary of an absorbing solid.
///
/// The mesh **must be closed**. Every ray that enters has to leave, or the
/// front and back contributions do not cancel and the pixel accumulates a
/// nonsense thickness. An open mesh does not fail loudly; it renders as though
/// the missing faces were infinitely far away. The alpha channel of the moment
/// target counts faces precisely so that this shows up — see
/// [`prepare::MomentTexture`].
#[derive(Component, Debug, Clone, Copy)]
pub struct MomentVolume {
    /// Absorbance per world unit of path length. Higher is more solid.
    pub sigma: f32,
    /// Linear RGB, and a *transmission* rather than a surface colour: this is
    /// the fraction of each channel the medium lets through. A tint of
    /// `(0, 0, 1)` absorbs red and green completely and so reads blue.
    ///
    /// Kept per channel because absorbance is per channel anyway — the three
    /// accumulate independently in the moment target at no extra cost, which is
    /// the coloured extinction of the reference document's §8 for free at
    /// `k = 0`.
    pub tint: Vec3,
}

impl Default for MomentVolume {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            tint: Vec3::new(0.45, 0.62, 0.85),
        }
    }
}

pub struct MomentPlugin;

impl Plugin for MomentPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "moment.wgsl");
        embedded_asset!(app, "resolve.wgsl");

        app.add_plugins(ExtractComponentPlugin::<MomentTransparency>::default())
            .add_systems(Startup, validate::spawn_test_scene)
            // Chained, so the deferred insert of `MomentTransparency` is
            // applied before the systems that key off it. Without the sync
            // point the depth usage is widened a frame late, and the first
            // frame fails validation because the bind group asks to sample a
            // write-only depth buffer.
            .add_systems(
                Update,
                (
                    validate::enable_on_cameras,
                    validate::warn_about_msaa,
                    widen_depth_usage,
                )
                    .chain(),
            );
    }

    /// Installed in `finish` rather than `build` because the decision needs a
    /// `RenderDevice`, and that does not exist until the render app has been
    /// initialised.
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Additive blending into a 32-bit float target is what the whole method
        // rests on, and it is not in the WebGPU baseline. Refusing here leaves
        // the volumes undrawn rather than silently wrong, which is what the
        // reference document's §9 asks for. Bevy already requests every feature
        // the adapter supports, so there is nothing to enable — only to check.
        let device = render_app.world().resource::<RenderDevice>();
        if !device.features().contains(WgpuFeatures::FLOAT32_BLENDABLE) {
            warn!(
                "draw: moment transparency needs the FLOAT32_BLENDABLE feature, which this \
                 adapter does not have; absorbing volumes will not be drawn"
            );
            return;
        }

        render_app
            // The pipelines themselves need an `AssetServer` and the fullscreen
            // shader, so they are built in `RenderStartup`. Everything else here
            // is a plain default and is better declared than constructed.
            .init_resource::<extract::ExtractedVolumes>()
            .init_resource::<prepare::MomentInstances>()
            .init_resource::<pipeline::QueuedMomentPipelines>()
            .init_resource::<SpecializedMeshPipelines<pipeline::MomentMeshPipeline>>()
            .add_systems(RenderStartup, init_moment_pipelines)
            .add_systems(ExtractSchedule, extract::extract_volumes)
            .add_systems(
                Render,
                (
                    (prepare_moment_textures, prepare_moment_instances)
                        .in_set(RenderSystems::PrepareResources),
                    queue_moment_pipelines.in_set(RenderSystems::Queue),
                    prepare_moment_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // After the transparent pass, not merely after the opaque one.
            // The resolve dims whatever is already in the view target, so
            // anything drawn after it escapes absorption entirely — and
            // `Transparent3d` carries more than alpha-blended meshes. Gizmos go
            // through it too, so ordering only against the opaque pass left the
            // grid showing through the volumes at full brightness while
            // everything else behind them dimmed correctly.
            //
            // The accumulation pass has no such constraint: it reads the opaque
            // depth buffer, which the transparent pass does not write. It is
            // chained to the resolve purely to keep the pair adjacent.
            .add_systems(
                Core3d,
                (moment_pass, moment_resolve)
                    .chain()
                    .after(main_transparent_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// Asks for the depth texture to be readable as well as writable.
///
/// The moment pass clamps against opaque depth in the fragment shader rather
/// than depth-testing (see [`pass`]), so it needs to *sample* the buffer the
/// opaque pass produced. Without this the texture is created write-only and the
/// bind group is rejected.
///
/// Written as "set it if it is not set" rather than keyed on
/// `Changed<Camera3d>`, which is what `bevy_core_pipeline`'s OIT does. That
/// filter assumes the settings component arrives with the camera; here it can
/// arrive later, and then the camera never changes again and the usage is never
/// widened. The explicit test also means this does not touch `Camera3d` on
/// every pass, which would mark it changed every frame and defeat
/// [`crate::redraw`].
fn widen_depth_usage(mut cameras: Query<&mut Camera3d, With<MomentTransparency>>) {
    let wanted = TextureUsages::TEXTURE_BINDING.bits();
    for mut camera in &mut cameras {
        if camera.depth_texture_usages.0 & wanted == 0 {
            camera.depth_texture_usages.0 |= wanted;
        }
    }
}

use bevy::render::render_resource::TextureUsages;
