//! The default backend: moment-based order-independent transparency.
//!
//! Opaque geometry goes through Bevy's ordinary passes and is lit normally.
//! Anything that transmits does not blend: a closed mesh is treated as the
//! boundary of a solid absorbing light at a uniform rate, and a sampled grid as
//! a medium along the ray. What you see through them is the interior — thick
//! parts read dark, thin parts clear — and nested or overlapping ones compose
//! correctly whatever order they are drawn in.
//!
//! The two halves are the point. Absorbance accumulates up to the **opaque
//! depth**, so a density map in front of a ribbon dims it and one behind does
//! not, which is what lets a structure be shown inside the map it was built
//! from.
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
//! spike, so every moment of it has an antiderivative:
//!
//! ```text
//! dA/dw   = sigma * span   inside the mesh, 0 outside
//! F_k(w)  = sigma * span * w^(k+1) / (k+1)
//! ```
//!
//! where `w` is depth warped into `[0, 1]` across the bound in
//! [`prepare::MomentBounds`]. An interior interval contributes
//! `F_k(w_out) - F_k(w_in)`. Front faces therefore add `-F_k(w)` and back faces
//! add `+F_k(w)`, and the additive blend performs the pairing on its own.
//! Non-convex and nested meshes need no special handling, which is the whole
//! point.
//!
//! One draw does both signs: `cull_mode: None` and a branch on
//! `@builtin(front_facing)`.
//!
//! # What the moments buy
//!
//! `k = 0` alone gives the total absorbance along the ray, which is already
//! exact for any arrangement of pure absorbers in front of opaque geometry —
//! however tangled, because the opaque depth clamp truncates each interval in
//! the right place. Four moments describe *where along the ray* the absorbance
//! sits, so the resolve can ask for the absorbance in front of any depth
//! instead of only the total. That is what transparent geometry at an
//! intermediate depth needs, and what emission and in-scattering will need.
//!
//! # Where this is in the build order
//!
//! Steps 1 to 4 of `ref/mboit-bevy-reference.md` §11: signed thickness, an
//! analytic reference, four power moments, and nested meshes. The render-world
//! half of those was built and validated against the closed form for a sphere
//! before the backend seam existed, and is ported here rather than rederived.
//! The check that validated it — a sphere of known absorbance rendered into an
//! orthographic camera of its own and compared per pixel against
//! `exp(-sigma * 2*sqrt(r^2 - d^2))` — has been removed; recover it from
//! history if a later step needs a reference image to diff against.
//!
//! Not yet done: a non-linear warp (step 5 — the warp here is linear, which is
//! the cheapest polynomial that keeps `F_k` closed form), light-space moments
//! (step 6), in-scattering (step 7), and trigonometric moments (step 8). Nor
//! per-view culling, or batching through a real phase item.
//!
//! # What it draws
//!
//! Six kinds, split by whether they transmit.
//!
//! **Opaque, through the ordinary passes:** `points`, `ball-and-stick`,
//! `glycan`, and `cartoon` in its default mode. These write depth and are the
//! thing the absorbance is measured *in front of*.
//!
//! **Transmitting, into the moment buffer:** `surface`, `volume`, and `cartoon`
//! in `absorbing` mode.
//!
//! All are `shared: true`: a closed triangle mesh or a ribbon is the same
//! physical thing whichever pathway draws it, so [`rt`](super::rt) may register
//! the same ids. A kind a pathway cannot do is simply absent from the registry
//! and refused by name rather than drawn wrongly.
//!
//! An opaque kind carries **no** [`MomentVolume`], which means
//! [`place_volumes`] does not match it — each has its own placement system
//! instead. Anything opaque added later needs the same pair.
//!
//! See `ref/mboit-bevy-reference.md`. Two of its notes do not survive contact
//! with Bevy 0.19 and are corrected here — see [`pass`] for the depth handling
//! and [`pipeline`] for the depth comparison.

use bevy::asset::embedded_asset;
use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::{
    SpecializedMeshPipelines, SpecializedRenderPipelines, TextureUsages,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::load_shader_library;

use crate::scene::link::Placement;
use crate::scene::registry::ActorRegistry;

mod cartoon;
mod extract;
mod glycan;
mod molecule;
mod pass;
mod points;
mod pipeline;
mod prepare;
mod surface;
mod volume;

// Re-exported rather than imported separately in each kind module, exactly as
// the raytracing pathway does it: whether a helper is shared or belongs to this
// pathway is a fact about the pathway, not something each kind should track.
pub(crate) use super::{Actor, Dirty, Draw, Invalidate, Place, bound, mark};

use pass::{grid_emit_pass, moment_pass, moment_resolve, shell_pass};
use pipeline::{
    init_moment_pipelines, queue_grid_pipelines, queue_moment_pipelines, queue_shell_pipelines,
};
use prepare::{
    prepare_grid_bind_groups, prepare_moment_bind_groups, prepare_moment_instances,
    prepare_moment_textures, prepare_shell_bind_groups, prepare_shell_lighting,
};

/// Marks a view the moment passes run on.
///
/// Not an opt-in. A backend is a whole pathway chosen once at launch, so if
/// this one is running then every 3D view is drawn by it — [`moment_cameras`]
/// puts this on each of them and nothing ever asks per camera. It exists
/// because the render world still needs to be told *which* views to allocate
/// moment targets for, which is a different question from which views want
/// them.
#[derive(Component, Debug, Clone, Copy, Default, ExtractComponent)]
pub struct MomentView;

/// How a mesh deposits absorbance into the moment buffer.
///
/// Two of the three content types in `ref/mboit-bevy-reference.md` §3, and the
/// choice between them is a choice about what the mesh *is* rather than a
/// quality setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Depiction {
    /// The mesh is the boundary of a solid that absorbs throughout — §3.3.
    ///
    /// The measure has a density, so each fragment contributes the
    /// antiderivative evaluated at its own depth and the additive blend pairs
    /// front with back. Thickness is what you see: a thin part reads clear and
    /// a thick one dark, which is the whole reason to prefer it when the mesh
    /// allows.
    ///
    /// The mesh **must be closed**. Every ray that enters has to leave, or the
    /// contributions do not cancel. An open mesh does not fail loudly: the
    /// unpaired front face drives the absorbance negative, the resolve clamps
    /// it away, and the part reads *too clear* rather than too dark.
    Interior {
        /// Absorbance per world unit of path length.
        sigma: f32,
    },
    /// The mesh is a film, and each fragment is a spike of absorbance at its
    /// own depth — §3.1.
    ///
    /// This is the classic MBOIT surface formulation, and its virtue here is
    /// what it does *not* require: a spike needs no closing face, so an open
    /// shell, a single triangle and a self-intersecting soup are all equally
    /// valid. That makes it the depiction for arbitrary geometry — CAD
    /// tessellations especially, which are routinely not closed.
    ///
    /// What it gives up is thickness. Every crossing costs the same whether the
    /// part is a millimetre or a metre through, so the picture reads as stacked
    /// tinted sheets rather than as a solid.
    Film {
        /// Already `-ln(1 - alpha)`, converted where the opacity is set rather
        /// than per fragment. The logarithm diverges at 1, so the clamp that
        /// keeps it finite belongs with the control a person moves.
        absorbance: f32,
    },
}

/// Marks a mesh as something the moment passes should accumulate.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct MomentVolume {
    /// Which of the two ways of depositing absorbance this mesh uses.
    pub depiction: Depiction,
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
            depiction: Depiction::Interior { sigma: 1.0 },
            tint: Vec3::new(0.45, 0.62, 0.85),
        }
    }
}

impl Depiction {
    /// The one number the shader needs, and which formula to use it in.
    ///
    /// Packed rather than branched on the CPU because both depictions share a
    /// pipeline: the difference is four lines of the fragment shader, not a
    /// different pass, and specialising over it would double the pipelines to
    /// save one comparison per fragment.
    fn packed(self) -> (f32, u32) {
        match self {
            Depiction::Interior { sigma } => (sigma.max(0.0), 0),
            Depiction::Film { absorbance } => (absorbance.max(0.0), 1),
        }
    }
}

/// Draws a volume's boundary a second time, as a thin dielectric shell.
///
/// The absorbing interior says how much light gets through; it says nothing
/// about the *surface*, so a volume alone reads like coloured fog with no
/// skin on it. This adds the one thing a dielectric boundary does that the
/// interior cannot: a specular reflection, Fresnel-weighted so it is faint
/// head-on and bright at grazing angles. That rim is most of what makes glass
/// read as glass.
///
/// **It transmits everything.** The shell absorbs nothing and blocks nothing —
/// the interior remains the only thing attenuating what is behind. So the pass
/// is purely additive, which also makes it order-independent like everything
/// else here: both faces of the shell contribute, in any order, and nothing has
/// to be sorted.
///
/// The energy it adds is not taken from what passes through, so a shell at
/// grazing incidence brightens the picture rather than redistributing it. A
/// physically closed account would remove the reflected fraction from the
/// transmitted one, which means the resolve would have to know about the shell.
/// That is a real coupling and is not worth it for a first pass.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct MomentShell {
    /// Reflectance straight on, which is what an index of refraction comes
    /// down to here: `((n - 1) / (n + 1))^2`. Glass is about 0.04.
    pub f0: f32,
    /// How tight the highlight is. 0 is a mirror, 1 is a sheen spread over the
    /// whole surface.
    pub roughness: f32,
}

/// The render-world half: the two passes and everything they need.
///
/// Split out from [`MomentBackendPlugin`] so it can be added without the
/// actor kinds — which is what lets the accumulation be exercised with no
/// actors, no gRPC and no interface in the way, the same isolation
/// `examples/solari_smoke.rs` gives the raytracer.
///
/// This pathway used to be called `experimental`, and the name survives in the
/// reference documents under `ref/`.
pub struct MomentRenderPlugin;

impl Plugin for MomentRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "moment.wgsl");
        embedded_asset!(app, "resolve.wgsl");
        embedded_asset!(app, "shell.wgsl");
        embedded_asset!(app, "volume.wgsl");
        embedded_asset!(app, "emit.wgsl");
        embedded_asset!(app, "point_quad.wgsl");
        // Imported by nothing yet: the fullscreen resolve applies the total
        // absorbance, which is exact at its own query point. This earns its
        // place when something asks at an intermediate depth. See the header of
        // the file itself, which records why more moments would not help.
        load_shader_library!(app, "reconstruct.wgsl");

        app.add_plugins(ExtractComponentPlugin::<MomentView>::default())
            // Chained, so the deferred insert of `MomentView` is applied before
            // the system that keys off it. Without the sync point the depth
            // usage is widened a frame late, and the first frame fails
            // validation because the bind group asks to sample a write-only
            // depth buffer.
            .add_systems(
                Update,
                (moment_cameras, widen_depth_usage).chain(),
            );
    }

    /// The render half is installed in `finish` because it needs the render app
    /// to exist. Whether the adapter can run it at all is not asked here —
    /// [`super::probe`] has already refused at startup if it cannot, which is
    /// what keeps the refusal loud and in one place.
    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // The pipelines themselves need an `AssetServer` and the fullscreen
            // shader, so they are built in `RenderStartup`. Everything else here
            // is a plain default and is better declared than constructed.
            .init_resource::<extract::ExtractedVolumes>()
            .init_resource::<extract::ExtractedGrids>()
            .init_resource::<extract::ExtractedShellLighting>()
            .init_resource::<prepare::MomentInstances>()
            .init_resource::<prepare::GridBindGroups>()
            .init_resource::<prepare::ShellLightingBuffer>()
            .init_resource::<pipeline::QueuedMomentPipelines>()
            .init_resource::<pipeline::QueuedShellPipelines>()
            .init_resource::<pipeline::QueuedGridPipelines>()
            .init_resource::<SpecializedMeshPipelines<pipeline::MomentMeshPipeline>>()
            .init_resource::<SpecializedMeshPipelines<pipeline::ShellMeshPipeline>>()
            .init_resource::<SpecializedRenderPipelines<pipeline::GridPipelines>>()
            .add_systems(RenderStartup, init_moment_pipelines)
            .add_systems(
                ExtractSchedule,
                (
                    extract::extract_volumes,
                    extract::extract_grids,
                    extract::extract_shell_lighting,
                ),
            )
            .add_systems(
                Render,
                (
                    (
                        prepare_moment_textures,
                        prepare_moment_instances,
                        prepare_shell_lighting,
                    )
                        .in_set(RenderSystems::PrepareResources),
                    (
                        queue_moment_pipelines,
                        queue_shell_pipelines,
                        queue_grid_pipelines,
                    )
                        .in_set(RenderSystems::Queue),
                    (
                        prepare_moment_bind_groups,
                        prepare_shell_bind_groups,
                        // After the queue, because the grid layout it builds
                        // against is owned by the pipeline resource.
                        prepare_grid_bind_groups,
                    )
                        .in_set(RenderSystems::PrepareBindGroups),
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
            // The shell comes last, after the resolve has dimmed the target.
            // A reflection off a glass surface never entered the medium, so it
            // must not be attenuated by it — see [`pass::shell_pass`].
            // The emission pass joins the shell after the resolve. Both add
            // light the medium never absorbed — a reflection off the near face
            // in one case, the volume's own glow in the other — so both must
            // escape the dimming the resolve applies. Their order between
            // themselves does not matter, because both blend additively.
            .add_systems(
                Core3d,
                (moment_pass, moment_resolve, grid_emit_pass, shell_pass)
                    .chain()
                    .after(main_transparent_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// The whole pathway: the passes, plus the actor kinds built for them.
pub struct MomentBackendPlugin;

impl Plugin for MomentBackendPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MomentRenderPlugin);

        // `points` draws through a custom material, which needs its own
        // `MaterialPlugin` to create the `Assets<PointQuadMaterial>` its draw
        // system asks for. Registering a kind is not enough: without this the
        // system fails parameter validation on the first frame a point cloud
        // exists, and takes the app with it.
        app.add_plugins(MaterialPlugin::<points::PointQuadMaterial>::default());

        {
            let mut registry = app.world_mut().resource_mut::<ActorRegistry>();
            surface::register(&mut registry);
            cartoon::register(&mut registry);
            glycan::register(&mut registry);
            molecule::register(&mut registry);
            points::register(&mut registry);
            volume::register(&mut registry);
        }

        app.add_systems(
            Update,
            (
                (
                    surface::invalidate,
                    cartoon::invalidate,
                    glycan::invalidate,
                    molecule::invalidate,
                    points::invalidate,
                    volume::invalidate,
                )
                    .in_set(Invalidate),
                (
                    surface::draw_surfaces,
                    cartoon::draw_cartoons,
                    glycan::draw_glycans,
                    molecule::draw_molecules,
                    points::draw_points,
                    volume::draw_volumes,
                )
                    .in_set(Draw),
                (
                    place_volumes,
                    cartoon::place_cartoons,
                    glycan::place_glycans,
                    molecule::place_molecules,
                    points::place_points,
                    volume::place_grids,
                )
                    .in_set(Place),
            ),
        );
    }
}

/// Marks every 3D camera as one this pathway draws.
///
/// Nothing else. It does not touch `Msaa`: the moment target takes whatever
/// sample count the view has, and the two passes are specialised to match — see
/// [`pipeline`]. Only the `rt` pathway has to force `Msaa::Off`, and it does
/// that in its own module rather than here.
fn moment_cameras(mut commands: Commands, cameras: Query<Entity, Added<Camera3d>>) {
    for camera in &cameras {
        commands.entity(camera).insert(MomentView);
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
fn widen_depth_usage(mut cameras: Query<&mut Camera3d, With<MomentView>>) {
    let wanted = TextureUsages::TEXTURE_BINDING.bits();
    for mut camera in &mut cameras {
        if camera.depth_texture_usages.0 & wanted == 0 {
            camera.depth_texture_usages.0 |= wanted;
        }
    }
}

/// Gives every placement of an actor the mesh and the absorbance that actor
/// owns.
///
/// The same job as the default backend's `copy_meshes`, and for the same
/// reason: an actor is hidden and holds the asset, a placement is a child of an
/// object and is what actually draws. [`extract::extract_volumes`] queries
/// `Mesh3d` and [`MomentVolume`] together, so a placement needs both or it is
/// not a volume at all.
///
/// Both are compared before writing, so a settled scene inserts nothing. The
/// absorbance has to be compared as well as the handle: dragging `sigma`
/// rewrites the actor's [`MomentVolume`] and never touches its mesh, and
/// keying on the handle alone left every placement absorbing at whatever rate
/// it was first drawn with.
#[allow(clippy::type_complexity)]
fn place_volumes(
    mut commands: Commands,
    actors: Query<(&Mesh3d, &MomentVolume, Option<&MomentShell>)>,
    placements: Query<(
        Entity,
        &Placement,
        Option<&Mesh3d>,
        Option<&MomentVolume>,
        Option<&MomentShell>,
    )>,
) {
    for (entity, placement, mesh3d, current, current_shell) in &placements {
        let Ok((mesh, volume, shell)) = actors.get(placement.0) else {
            // The actor has not been drawn yet. Nothing to copy, and it will be
            // here next frame.
            continue;
        };
        if mesh3d.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert(mesh.clone());
        }
        if current != Some(volume) {
            commands.entity(entity).insert(*volume);
        }
        // The shell too, or a placement draws the interior of a piece of glass
        // with no surface on it. Removed as well as added, so switching the
        // shell off takes it off every copy.
        if current_shell != shell {
            match shell {
                Some(shell) => commands.entity(entity).insert(*shell),
                None => commands.entity(entity).remove::<MomentShell>(),
            };
        }
    }
}
