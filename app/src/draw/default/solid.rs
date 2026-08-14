//! Closed triangle meshes, drawn as the boundary of an absorbing solid.
//!
//! Not the same thing as [`mesh`](super::mesh), which is an ordinary lit mesh.
//! This kind says the triangles *bound a medium*: what you see is the interior,
//! so thick parts read dark and thin parts clear, and overlapping or nested
//! solids compose without sorting.
//!
//! The two were one kind, and splitting them is what stopped a client binding
//! triangles and getting something it had not asked for. Same geometry, two
//! different claims about what it is, so two names.
//!
//! They are two kinds and not one kind with a mode, because they go through
//! different passes: `mesh` through Bevy's ordinary opaque pass with a
//! `StandardMaterial`, this one through [`moment_pass`](super::pass) and
//! [`shell_pass`](super::pass) with [`MomentVolume`](super::MomentVolume). A
//! mode inside a kind is right when the pass is the same — `solid` against
//! `film`, below, is exactly that — and wrong when it is not. Their parameters
//! are disjoint too: `sigma` means nothing to a lit mesh and `double_sided`
//! nothing to a medium.
//!
//! There is no `double_sided` here, unlike `mesh`: it is a lighting choice,
//! and both faces are always drawn — they *have* to be, since the two of them
//! are the endpoints of the interval being integrated.
//!
//! # Two ways of drawing, over each other
//!
//! An actor here can produce both halves of what a piece of glass looks like:
//!
//! - the **interior**, always — a solid absorbing at `sigma` per unit of path
//!   length, which is what makes thick parts read dark and thin parts clear;
//! - the **surface**, when `shell` is on — a thin dielectric skin adding a
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
//! # What is not read
//!
//! No per-vertex `colour`. Absorbance is a property of a medium, so it is one
//! value for the whole volume rather than something varying across a surface
//! the interior does not have; colour arrives through the `tint` parameter,
//! read as a transmission. `normals` *are* read, but only when a shell is on — the
//! accumulation cares where a boundary is, not which way it faces, so a volume
//! with no skin never pays for them.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::scene::registry::Bindings;
use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, flag, float, text};
use crate::scene::subset::Remap;
use crate::scene::{DataArray, DataStore, Dtype, Subset};

use super::{Actor, Depiction, Dirty, MomentShell, MomentVolume, bound, mark};

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
pub struct AbsorbingStyle {
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
    ParamSpec {
        id: "positions",
        label: "positions",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    ParamSpec {
        id: "indices",
        label: "triangles (must close the surface)",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint32],
            shape: &[0, 3],
            required: true,
            structural: true,
        },
    },
    // Unbound means "work them out from the triangles". Only the shell reads
    // them — an absorbance depends on where a boundary is, not which way it
    // faces — so a volume with no shell never pays for them.
    ParamSpec {
        id: "normals",
        label: "normals",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: false,
            structural: true,
        },
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
        id: "solid",
        label: "absorbing solid",
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
            entity.insert(AbsorbingStyle {
                depiction,
                tint: crate::draw::tint(params, "tint", Vec3::splat(0.8)),
                ior: flag(params, "shell", false).then(|| float(params, "ior", 1.49)),
                roughness: float(params, "roughness", 0.15),
            });
        },
    });
}

/// Turning the shell on or off is a *geometry* change, and the rest is not.
///
/// Everything here writes a number the render world reads per frame — except
/// switching the shell on, which needs normals the mesh may not carry. That is
/// a rebuild, and the alternative was building normals for every volume in case
/// a shell is switched on later.
pub fn invalidate(
    mut commands: Commands,
    changed: Query<(Entity, &AbsorbingStyle), Changed<AbsorbingStyle>>,
    mut previous: Local<bevy::platform::collections::HashMap<Entity, bool>>,
) {
    for (entity, style) in &changed {
        let wants_shell = style.ior.is_some();
        let had_shell = previous.insert(entity, wants_shell);
        if had_shell.is_some_and(|had| had != wants_shell) {
            mark(&mut commands, entity, Dirty::GEOMETRY);
        } else {
            mark(&mut commands, entity, Dirty::MATERIAL);
        }
    }
}

/// What this backend needs to redraw one actor.
///
/// The tail is what makes it this pathway's query rather than another's: a mesh
/// and an absorbance, which is what the moment passes consume. Carrying the
/// previous mesh handle is what makes reuse rather than reallocation possible,
/// so dragging `sigma` allocates nothing.
type Drawable<'a> = (
    Actor<'a, AbsorbingStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MomentVolume>,
);

pub fn draw_solids(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
) {
    for ((entity, style, subset, bindings, dirty), mesh3d, _volume) in &dirty {
        if !dirty.any() {
            continue;
        }

        if dirty.geometry
            && let Some(mesh) = build(bindings, subset, &store, &arrays, style.ior.is_some())
        {
            ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, mesh);
        }

        // Unconditional rather than gated on `dirty.material`: a rebuild leaves
        // the absorbance untouched, and an actor drawn for the first time is
        // dirty in every way at once, so writing it here costs one comparison
        // in `place_volumes` and cannot be missed.
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

/// Replaces a mesh in place when the actor already has one.
///
/// Reusing the handle keeps every placement pointing at the same asset — and,
/// more to the point, stops each rebuild leaking a fresh `Mesh` into `Assets`.
/// The early return rather than an `else` is load-bearing: it ends the mutable
/// borrow of `meshes` before the second arm needs one.
///
/// The same function as the default backend's, written again here. Backends
/// duplicate rather than share their drawing code by design — see
/// [`crate::draw`] — and this one is four lines.
fn ensure_mesh(
    commands: &mut Commands,
    entity: Entity,
    meshes: &mut Assets<Mesh>,
    existing: Option<&Mesh3d>,
    mesh: Mesh,
) {
    if let Some(Mesh3d(handle)) = existing
        && let Some(mut slot) = meshes.get_mut(handle)
    {
        *slot = mesh;
        return;
    }
    commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
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

/// Builds the boundary mesh from what the actor binds.
///
/// Positions and triangles only. Returns `None` when there is nothing to draw,
/// which covers an array released underneath a binding as well as a subset that
/// left no whole triangle.
fn build(
    bindings: &Bindings,
    subset: &Subset,
    store: &DataStore,
    arrays: &Assets<DataArray>,
    needs_normals: bool,
) -> Option<Mesh> {
    let (position_array, index_array) = (
        bound(bindings, "positions", store, arrays)?,
        bound(bindings, "indices", store, arrays)?,
    );
    let all = position_array.to_vec3();
    let Some(all_indices) = index_array.to_u32() else {
        warn!("draw: mesh indices are not an integer type");
        return None;
    };
    if all.is_empty() || all_indices.is_empty() {
        return None;
    }
    if let Some(out_of_range) = all_indices.iter().find(|i| **i as usize >= all.len()) {
        warn!(
            "draw: mesh index {out_of_range} exceeds {} vertices",
            all.len()
        );
        return None;
    }

    // A triangle survives only if all three of its corners do, and the
    // surviving points are renumbered, so the connectivity has to be rewritten
    // rather than merely filtered. Worth saying plainly: a subset of a closed
    // mesh is generally *not* closed, and this pathway will draw the result as
    // though the cut faces were infinitely far away.
    let kept = subset.selected(all.len(), arrays);
    let (positions, indices) = match &kept {
        Some(kept) => {
            let remap = Remap::new(kept, all.len());
            let positions: Vec<Vec3> = kept.iter().map(|index| all[*index as usize]).collect();
            let indices: Vec<u32> = all_indices
                .chunks_exact(3)
                .filter_map(|corners| remap.cell(corners))
                .flatten()
                .collect();
            if indices.is_empty() {
                info!("draw: a subset left no whole triangles; nothing to draw");
                return None;
            }
            (positions, indices)
        }
        None => (all, all_indices),
    };

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        positions
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect::<Vec<_>>(),
    );
    mesh.insert_indices(Indices::U32(indices));

    // Only when a shell is on. Normals are dead weight in the accumulation —
    // they widen every vertex by twelve bytes for a pass that never reads them
    // — so a volume with no skin does not carry them.
    if needs_normals {
        let supplied = bound(bindings, "normals", store, arrays)
            .map(|array| array.to_vec3())
            .filter(|normals| normals.len() == position_array.count() as usize)
            .map(|normals| match &kept {
                Some(kept) => kept.iter().map(|index| normals[*index as usize]).collect(),
                None => normals,
            });
        match supplied {
            Some(normals) => mesh.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                normals.iter().map(|n| [n.x, n.y, n.z]).collect::<Vec<_>>(),
            ),
            // Smooth, because the mesh is indexed and a shell is meant to read
            // as a continuous surface. Faceting a glass object turns one
            // highlight into a mosaic of them.
            None => mesh.compute_normals(),
        }
    }

    debug!(
        "draw: absorbing surface rebuilt with {} vertices",
        positions.len()
    );
    Some(mesh)
}
