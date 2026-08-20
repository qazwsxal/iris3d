//! Triangles drawn as the boundary of an absorbing medium.
//!
//! Not the same thing as [`surface`](super::surface), which draws them as an
//! opaque lit skin. This kind says the triangles *bound a medium*: what you see
//! is the interior, so thick parts read dark and thin parts clear, and
//! overlapping or nested bodies compose without sorting. Same geometry, two
//! different claims about what it is, so two kinds — the rule is the pass, and
//! their parameters are disjoint.
//!
//! An actor here draws two things over each other: the **interior** always, a
//! body absorbing at `sigma` per unit of path length; and the **boundary** when
//! `shell` is on, a thin dielectric skin that reflects and absorbs nothing. The
//! shell is what stops the shape reading as coloured fog. See [`MomentShell`].
//!
//! `mode` decides whether the mesh must be closed. In `solid` it must — every
//! ray entering the interior has to leave it, or the contributions do not
//! cancel. In `film` it need not, because each fragment is a spike at its own
//! depth and needs no partner; that is the mode for geometry you did not author,
//! CAD tessellations especially. What `film` costs is thickness.
//!
//! Per-vertex colours are ignored here and normals are read only when a shell is
//! on. Both are attributes of a mesh this kind shares rather than owns, so it
//! cannot decline to carry them — the same geometry drawn as a lit
//! [`surface`](super::surface) wants exactly the ones this pass ignores.
//!
//! **`docs/design/moment-transparency.md` has the rest:** why transparency is a
//! kind here when ParaView, PyMOL and ChimeraX all make it an opacity slider,
//! where the name comes from, and why closedness is not checked.

use bevy::prelude::*;

use iris3d_model::{ParamKind, ParamSpec, flag, float, text};
use iris3d_scene::DataStore;
use iris3d_scene::registry::{ActorKind, ActorRegistry};

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
    crate::TINT,
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
                tint: crate::tint(params, "tint", Vec3::splat(0.8)),
                ior: flag(params, "shell", false).then(|| float(params, "ior", 1.49)),
                roughness: float(params, "roughness", 0.15),
            });
        },
    });
}

/// Nothing here changes the mesh: this kind does not own one.
///
/// The geometry is shared and carries whatever it carries, so switching the
/// shell on can only succeed or be refused by the pipeline. There is nothing to
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

pub fn draw_media(mut commands: Commands, store: Res<DataStore>, dirty: Query<Drawable>) {
    for ((entity, style, bindings, dirty), mesh3d, _volume) in &dirty {
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
/// Shared with `cartoon` rather than written twice. Backends
/// duplicate their drawing code from *each other* by design — see
/// [`crate`] — but this is one pathway's own physics, and both of its
/// kinds mean exactly the same thing by an index of refraction.
pub fn normal_reflectance(ior: f32) -> f32 {
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
/// Already linear when it gets here: `crate::tint` converts once, where
/// the parameter is read, because the accumulation works in linear light
/// throughout.
///
/// Shared with `cartoon`; see [`normal_reflectance`] for why
/// that is not a breach of the rule that backends duplicate.
pub fn transmission(tint: Vec3) -> Vec3 {
    tint.clamp(Vec3::ZERO, Vec3::ONE)
}
