//! Triangles drawn as the boundary of an absorbing medium.
//!
//! Not the same thing as [`surface`](super::surface), which draws them as an
//! opaque lit skin. This kind says the triangles *bound a medium*: what you see
//! is the interior, so thick parts read dark and thin parts clear, and
//! overlapping or nested bodies compose without sorting.
//!
//! The two were one kind, and splitting them is what stopped a client binding
//! triangles and getting something it had not asked for. Same geometry, two
//! different claims about what it is, so two names.
//!
//! # The name
//!
//! A **medium** in the physical sense — light passing through it is absorbed
//! along the way. Blender calls the same thing a volume absorption shader on an
//! object's interior, and requires the same closed manifold mesh for it; the
//! word `volume` is spent here on [`volume`](super::volume), the grid actor, so
//! `medium` is what is left and what the physics calls it.
//!
//! It was called `solid` for a while, which was backwards twice over. In
//! ChimeraX `solid` is the **opaque** filled style, the opposite of this, and in
//! ordinary English a solid sounds like something you cannot see through. The
//! word survives where it is right — as this kind's `mode`, where `solid` means
//! a body with thickness against `film`, a surface without one.
//!
//! # Transparency is a kind here, and a property everywhere else
//!
//! Worth knowing, because it looks like a mistake. ParaView, PyMOL and ChimeraX
//! all make transparency an *opacity setting* on the ordinary surface
//! representation; none of them has a separate kind for it.
//!
//! This is not that. An opacity slider blends a surface with what is behind it;
//! this integrates absorbance along the path *through* a body, which is a
//! different physical claim, needs a closed mesh, and is the thing iris3d exists
//! to compose correctly against a volume. Blender agrees it is a different
//! thing rather than a slider.
//!
//! So they are two kinds and not one kind with a mode, and the rule is the pass:
//! `surface` goes through Bevy's ordinary opaque pass with a `StandardMaterial`,
//! this one through [`moment_pass`](super::pass) and [`shell_pass`](super::pass)
//! with [`MomentVolume`](super::MomentVolume). A mode inside a kind is right
//! when the pass is the same — `solid` against `film`, below, is exactly that —
//! and wrong when it is not. Their parameters are disjoint too: `sigma` means
//! nothing to a lit surface and `double_sided` nothing to a medium.
//!
//! There is no `double_sided` here, unlike `surface`: it is a lighting choice,
//! and both faces are always drawn — they *have* to be, since the two of them
//! are the endpoints of the interval being integrated.
//!
//! # Two ways of drawing, over each other
//!
//! An actor here can produce both halves of what a piece of glass looks like:
//!
//! - the **interior**, always — a body absorbing at `sigma` per unit of path
//!   length, which is what makes thick parts read dark and thin parts clear;
//! - the **boundary**, when `shell` is on — a thin dielectric skin adding a
//!   Fresnel-weighted specular reflection and absorbing nothing.
//!
//! They are two passes over the same mesh, and the second is what the shape
//! needs to stop reading as coloured fog. See [`MomentShell`].
//!
//! # Whether the mesh must be closed depends on `mode`
//!
//! In `solid` it must. Every ray entering the interior has to leave it, or the
//! contributions do not cancel; the pathway cannot tell an open mesh from a
//! closed one, because closedness is a property of the connectivity a client
//! uploads and checking it would cost a pass over every edge on every rebuild
//! to report a fact the client already knows.
//!
//! In `film` it need not. Each fragment is a spike at its own depth and needs
//! no partner, so an open shell, a lone triangle or a self-intersecting soup
//! are all valid. That is the mode for geometry you did not author — CAD
//! tessellations especially, which are routinely not closed. What it costs is
//! thickness: every crossing counts the same however deep the part is.
//!
//! # What it does not read
//!
//! The geometry's per-vertex colours mean nothing here. Absorbance is a property
//! of a medium, so it is one value for the whole volume rather than something
//! varying across a surface the interior does not have; colour arrives through
//! the `tint` parameter, read as a transmission. Its **normals** are read, but
//! only when a shell is on — the accumulation cares where a boundary is, not
//! which way it faces, so a volume with no skin never pays for them.
//!
//! Both are attributes of a mesh this kind shares rather than owns, so neither
//! is something it can decline to carry: the same geometry drawn as a lit
//! [`surface`](super::surface) wants exactly the ones this pass ignores. What
//! that costs is stride, and what it buys is one upload instead of two. The
//! accumulation pipeline pulls only the position out of whatever layout it is
//! given — see [`MomentMeshPipeline`](super::pipeline::MomentMeshPipeline).

use bevy::prelude::*;

use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, flag, float, text};
use crate::scene::DataStore;

use super::{Actor, Depiction, Dirty, MomentShell, MomentVolume, mark};

/// The two ways this kind can deposit absorbance. See [`Depiction`].
const MODES: &[&str] = &["solid", "film"];

/// How the mesh absorbs, and whether it has a skin.
///
/// Separate from [`MomentVolume`] even though it holds the same numbers,
/// because they answer to different owners: this is derived from the actor's
/// parameters by `apply`, and [`MomentVolume`] is the drawable output the render
/// world extracts. Keeping them apart is what lets one be edited by a client
/// while the other is rebuilt from it.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct MediumStyle {
    /// Already resolved from `mode` plus whichever of `sigma` and `alpha` that
    /// mode reads, so nothing downstream has to consult two parameters to know
    /// what one mesh is.
    pub depiction: Depiction,
    /// Index of refraction of the shell, or `None` for no shell at all.
    ///
    /// An option rather than a `shell: bool` beside an `ior` that means nothing
    /// when it is false. The two parameters on the wire stay separate because a
    /// client wants a checkbox and a slider, but by the time a kind reads them
    /// the impossible state is gone.
    pub ior: Option<f32>,
    pub roughness: f32,
    /// Linear RGB, read as a transmission. See [`transmission`].
    pub tint: Vec3,
}

const PARAMS: &[ParamSpec] = &[
    // The same one input `surface` takes, and the same handle in practice: two
    // actors of the two kinds over one geometry is what this whole split is for.
    // In `solid` mode the triangles must close the surface — see above.
    ParamSpec {
        id: "geometry",
        label: "geometry (must close the surface in solid mode)",
        kind: ParamKind::Geometry { required: true },
    },
    // Which of the two ways of depositing absorbance the mesh uses. `solid`
    // needs a closed mesh and shows thickness; `film` needs nothing and does
    // not. See `Depiction`.
    ParamSpec {
        id: "mode",
        label: "mode",
        kind: ParamKind::Choice {
            options: MODES,
            default: "solid",
        },
    },
    // Logarithmic, because the useful range is wide and the interesting part is
    // at the bottom of it: the difference between 0.05 and 0.1 is a volume you
    // can see through and one you can nearly see through, while 10 and 20 are
    // both simply opaque. Read only in `solid` mode.
    ParamSpec {
        id: "sigma",
        label: "absorbance per unit (solid)",
        kind: ParamKind::Float {
            default: 1.0,
            min: 0.001,
            max: 50.0,
            logarithmic: true,
        },
    },
    // Read only in `film` mode. Capped below 1 because the absorbance of a
    // fragment is `-ln(1 - alpha)`, which diverges there — the clamp belongs
    // with the control a person moves rather than in the shader.
    ParamSpec {
        id: "alpha",
        label: "opacity per surface (film)",
        kind: ParamKind::Float {
            default: 0.30,
            min: 0.001,
            max: 0.99,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "shell",
        label: "glass shell",
        kind: ParamKind::Bool { default: false },
    },
    // Acrylic by default, at 1.49. Glass is 1.52 and the two are very nearly
    // indistinguishable by this control alone — 0.039 reflectance straight on
    // against 0.043 — so what separates them by eye is the grazing rim, which
    // `shell.wgsl` caps rather than letting it reach a full mirror.
    //
    // 1.0 is vacuum and reflects nothing, which is why the shell is switched on
    // separately rather than by winding this down: "no shell" and "a shell that
    // happens to be invisible" cost different amounts.
    ParamSpec {
        id: "ior",
        label: "index of refraction",
        kind: ParamKind::Float {
            default: 1.49,
            min: 1.0,
            max: 3.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "roughness",
        label: "roughness",
        kind: ParamKind::Float {
            default: 0.15,
            min: 0.0,
            max: 1.0,
            logarithmic: false,
        },
    },
    crate::draw::TINT,
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "medium",
        label: "medium",
        params: PARAMS,
        apply: |entity, params| {
            let depiction = match text(params, "mode", "solid") {
                // `-ln(1 - alpha)` here rather than per fragment: it is one
                // logarithm per actor against one per pixel per surface, and
                // the clamp that keeps it finite sits next to the control that
                // could otherwise drive it to infinity.
                "film" => Depiction::Film {
                    absorbance: -(1.0 - float(params, "alpha", 0.30).clamp(0.001, 0.99)).ln(),
                },
                _ => Depiction::Interior {
                    sigma: float(params, "sigma", 1.0),
                },
            };
            entity.insert(MediumStyle {
                depiction,
                tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
                ior: flag(params, "shell", false).then(|| float(params, "ior", 1.49)),
                roughness: float(params, "roughness", 0.15),
            });
        },
    });
}

/// Nothing here changes the mesh, because this kind no longer owns one.
///
/// Turning the shell on used to be a rebuild: it needs normals, and a volume
/// with no shell was built without them to save the stride. The geometry is
/// shared now and carries whatever it carries, so switching the shell on can
/// only succeed or be refused by the pipeline — there is nothing left to
/// rebuild, and no way to add normals to a mesh another actor is drawing.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<MediumStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::MATERIAL);
    }
}

/// What this backend needs to redraw one actor.
///
/// The tail is what makes it this pathway's query rather than another's: a mesh
/// and an absorbance, which is what the moment passes consume.
type Drawable<'a> = (
    Actor<'a, MediumStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MomentVolume>,
);

pub fn draw_media(
    mut commands: Commands,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, _subset, bindings, dirty), mesh3d, _volume) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Some(geometry) = super::surface::geometry(bindings, &store) else {
            continue;
        };

        // The same handle a `mesh` actor over the same geometry holds. One
        // upload, two materials, two passes.
        if mesh3d.map(|Mesh3d(handle)| handle.id()) != Some(geometry.handle.id()) {
            commands
                .entity(entity)
                .insert(Mesh3d(geometry.handle.clone()));
        }

        // Said once, where the geometry is known, rather than left to a
        // pipeline error that names a missing vertex attribute without saying
        // which pass wanted it.
        if style.ior.is_some() && !geometry.meta.normals {
            warn!(
                "draw: a medium's shell needs normals, and \"{}\" carries none",
                geometry.meta.name
            );
        }

        // Unconditional rather than gated on `dirty.material`: an actor drawn
        // for the first time is dirty in every way at once, so writing it here
        // costs one comparison in `place_volumes` and cannot be missed.
        let mut actor = commands.entity(entity);
        actor.insert(MomentVolume {
            depiction: style.depiction,
            tint: transmission(style.tint),
        });
        match style.ior {
            Some(ior) => actor.insert(MomentShell {
                f0: normal_reflectance(ior),
                roughness: style.roughness.clamp(0.0, 1.0),
            }),
            // Removed rather than left with a zero reflectance, so the render
            // world skips the volume entirely instead of drawing a second pass
            // that contributes nothing.
            None => actor.remove::<MomentShell>(),
        };
    }
}

/// What fraction a dielectric reflects straight on, from its index of
/// refraction: the Fresnel equations at normal incidence, against air.
///
/// Glass at 1.5 gives 0.04, which is the number every renderer hardcodes for
/// "dielectric". Taking the index instead means water, ice and diamond are the
/// same control rather than three magic constants.
///
/// Shared with [`cartoon`](super::cartoon) rather than written twice. Backends
/// duplicate their drawing code from *each other* by design — see
/// [`crate::draw`] — but this is one pathway's own physics, and both of its
/// kinds mean exactly the same thing by an index of refraction.
pub(super) fn normal_reflectance(ior: f32) -> f32 {
    let ratio = (ior - 1.0) / (ior + 1.0);
    ratio * ratio
}

/// The flat colour, read as the fraction of each channel the solid lets
/// through.
///
/// `tint` is the only colour this kind has: absorbance is a property of the
/// medium, so it is one value for the whole volume rather than something that
/// varies across a surface it does not have. Reading it as a transmission is
/// what makes it intuitive — the volume shows in the colour it passes — and it
/// is also the only reading that is physically a tint rather than a paint.
///
/// Already linear when it gets here: `crate::draw::tint` converts once, where
/// the parameter is read, because the accumulation works in linear light
/// throughout.
///
/// Shared with [`cartoon`](super::cartoon); see [`normal_reflectance`] for why
/// that is not a breach of the rule that backends duplicate.
pub(super) fn transmission(tint: Vec3) -> Vec3 {
    tint.clamp(Vec3::ZERO, Vec3::ONE)
}
