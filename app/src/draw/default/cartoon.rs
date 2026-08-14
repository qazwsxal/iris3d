//! Cartoon ribbons, drawn as the boundary of an absorbing solid.
//!
//! `shared: true`, because a ribbon through a backbone is one physical thing
//! whether it is lit or integrated through — [`rt`](crate::draw::rt) can
//! register the same id whenever it grows the biology kinds. The curve and the
//! sweep are [`crate::draw::cartoon`]; what is here is the mapping onto the
//! moment passes.
//!
//! # Opaque is the point
//!
//! The `mode` here is not [`solid`](super::solid)'s. That one chooses
//! between two ways of *absorbing*, because a client uploads triangles and only
//! the client knows whether they close; this one chooses whether the ribbon
//! absorbs at all.
//!
//! **`opaque`**, the default, is what this pathway is for. The ribbon goes
//! through Bevy's ordinary opaque pass — lit, writing depth — and the moment
//! accumulation truncates every interval at that depth. So a volume in front of
//! the ribbon dims it by exactly the absorbance along the path from the eye to
//! the surface, and a volume behind it contributes nothing. That is what makes
//! the composition this backend exists for possible: a protein cartoon sitting
//! inside the electron-density map it was built from, with the density in front
//! blocking the light coming off the ribbon behind, correct from every angle and
//! with nothing sorted.
//!
//! Ordinary alpha blending cannot do that. The density is a participating medium
//! sampled along the ray, not a surface, so there is no single depth to sort it
//! against — and the ribbon threads through it, in front of some samples and
//! behind others, at every pixel.
//!
//! **`absorbing`** makes the ribbon a medium of its own instead: a solid whose
//! thickness you see through. Worth keeping for looking at one cartoon through
//! another, and it needs no closure check — the sweep caps every run, so unlike
//! an uploaded mesh this kind *knows* it is closed.
//!
//! # Colour depends on the mode
//!
//! An **opaque** ribbon takes vertex colours from a bound `colour` array, per
//! residue — colour by chain, by B-factor,
//! by anything. It is an ordinary lit mesh, and nothing about the moment passes
//! changes that.
//!
//! An **absorbing** one does not. Absorbance is a property of a medium, so it is
//! one value for the whole ribbon rather than something varying across a surface
//! the interior does not have; colour then comes from [`ColorBy::flat`] alone,
//! read as a transmission.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::draw::cartoon::{self, Ribbon, Style};
use crate::scene::link::Placement;
use crate::scene::registry::{
    ActorKind, ActorRegistry, Bindings, ParamKind, ParamSpec, flag, float, text,
};
use crate::scene::{ColorBy, DataArray, DataStore, Dtype, Subset};

use super::solid::{normal_reflectance, transmission};
use super::{Actor, Depiction, Dirty, MomentShell, MomentVolume, mark};

/// How the ribbon takes part in the frame.
///
/// Not a quality setting: it decides whether the cartoon is something light
/// passes *through* or something light *stops at*, which are different physical
/// claims about the same shape.
const MODES: &[&str] = &["opaque", "absorbing"];

const PARAMS: &[ParamSpec] = &[
    // Which of the two the ribbon is. `opaque` is the default because it is what
    // a cartoon usually means: the molecule is the subject and the volume around
    // it is the evidence, so the ribbon should stop light and the density should
    // dim it.
    ParamSpec {
        id: "mode",
        label: "mode",
        kind: ParamKind::Choice {
            options: MODES,
            default: "opaque",
        },
    },
    ParamSpec {
        id: "positions",
        label: "atom centres",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
        },
    },
    ParamSpec {
        id: "residue_index",
        label: "residue per atom",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32, Dtype::Uint64],
            shape: &[0],
            required: true,
        },
    },
    // The two halves of a dictionary-encoded name column, which is how text
    // travels: once per distinct value, never once per atom.
    ParamSpec {
        id: "atom_name_index",
        label: "atom name per atom",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32],
            shape: &[0],
            required: true,
        },
    },
    ParamSpec {
        id: "atom_name",
        label: "distinct atom names",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Str],
            shape: &[0],
            required: true,
        },
    },
    ParamSpec {
        id: "residue_sse",
        label: "secondary structure per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint8],
            shape: &[0],
            required: false,
        },
    },
    ParamSpec {
        id: "residue_chain_index",
        label: "chain per residue",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Uint16, Dtype::Uint32, Dtype::Uint64],
            shape: &[0],
            required: false,
        },
    },
    // Read in `opaque` only. An absorbing ribbon is a medium, and a medium has
    // one absorbance rather than a colour that varies across a surface it does
    // not have; an opaque one is an ordinary lit mesh and takes vertex colours
    // like any other. Declared unconditionally because a kind's parameters are
    // a fixed list — the mode decides whether it is *used*, and the doc says so.
    ParamSpec {
        id: "colour",
        label: "colour by (opaque only)",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
        },
    },
    ParamSpec {
        id: "size_factor",
        label: "half thickness (Å)",
        kind: ParamKind::Float {
            default: 0.2,
            min: 0.02,
            max: 1.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "aspect_ratio",
        label: "width / thickness",
        kind: ParamKind::Float {
            default: 5.0,
            min: 1.0,
            max: 15.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "nucleic_aspect_ratio",
        label: "nucleic width / thickness",
        kind: ParamKind::Float {
            default: 8.0,
            min: 1.0,
            max: 20.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "arrow_factor",
        label: "arrowhead width",
        kind: ParamKind::Float {
            default: 1.5,
            min: 1.0,
            max: 3.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "linear_segments",
        label: "samples per residue",
        kind: ParamKind::Float {
            default: 8.0,
            min: 2.0,
            max: 24.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "radial_segments",
        label: "sides of a round profile",
        kind: ParamKind::Float {
            default: 16.0,
            min: 3.0,
            max: 32.0,
            logarithmic: false,
        },
    },
    ParamSpec {
        id: "tubular_helices",
        label: "helices as tubes",
        kind: ParamKind::Bool { default: false },
    },
    // The ladder rungs. Both the block and its stick are closed solids like
    // everything else the sweep makes, so they absorb correctly rather than
    // needing the film depiction.
    ParamSpec {
        id: "base_rings",
        label: "nucleic base rings",
        kind: ParamKind::Bool { default: true },
    },
    // A ribbon is thin — 0.4 Å through at the default thickness — so the useful
    // absorbance is far higher than a CAD part's. At sigma 1 a default cartoon
    // is very nearly invisible, which is why this starts an order of magnitude
    // up from `surface`'s.
    ParamSpec {
        id: "sigma",
        label: "absorbance per unit",
        kind: ParamKind::Float {
            default: 8.0,
            min: 0.01,
            max: 200.0,
            logarithmic: true,
        },
    },
    ParamSpec {
        id: "shell",
        label: "glass shell",
        kind: ParamKind::Bool { default: false },
    },
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
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "cartoon",
        label: "cartoon",
        shared: true,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(CartoonStyle {
                geometry: Style {
                    size_factor: float(params, "size_factor", 0.2),
                    aspect_ratio: float(params, "aspect_ratio", 5.0),
                    nucleic_aspect_ratio: float(params, "nucleic_aspect_ratio", 8.0),
                    arrow_factor: float(params, "arrow_factor", 1.5),
                    linear_segments: float(params, "linear_segments", 8.0).round() as usize,
                    radial_segments: float(params, "radial_segments", 16.0).round() as usize,
                    tubular_helices: flag(params, "tubular_helices", false),
                    base_rings: flag(params, "base_rings", true),
                },
                opaque: text(params, "mode", "opaque") != "absorbing",
                sigma: float(params, "sigma", 8.0),
                ior: flag(params, "shell", false).then(|| float(params, "ior", 1.49)),
                roughness: float(params, "roughness", 0.15),
            });
        },
    });
}

/// The ribbon's shape, which of the two things it is, and how it absorbs if it
/// absorbs at all.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CartoonStyle {
    pub geometry: Style,
    /// Stop light rather than absorb it. See [`draw_cartoons`].
    pub opaque: bool,
    pub sigma: f32,
    /// Index of refraction of the shell, or `None` for no shell at all. See
    /// [`AbsorbingStyle::ior`](super::solid::AbsorbingStyle::ior) for why this
    /// is an option rather than a flag beside a number.
    pub ior: Option<f32>,
    pub roughness: f32,
}

/// Geometry changes rebuild; absorbance does not.
///
/// Unlike [`solid`](super::solid), switching the shell on is **not** a
/// rebuild here. The sweep produces normals whatever happens — they fall out of
/// the profile — so unlike an uploaded mesh there is nothing to go back and
/// compute. Changing `mode` *is* a rebuild, because the two depictions do not
/// want the same vertex attributes.
pub fn invalidate(
    mut commands: Commands,
    changed: Query<(Entity, &CartoonStyle), Changed<CartoonStyle>>,
    mut previous: Local<bevy::platform::collections::HashMap<Entity, (Style, bool)>>,
) {
    for (entity, style) in &changed {
        let now = (style.geometry, style.opaque);
        let was = previous.insert(entity, now);
        if was.is_some_and(|was| was == now) {
            mark(&mut commands, entity, Dirty::MATERIAL);
        } else {
            mark(&mut commands, entity, Dirty::GEOMETRY);
        }
    }
}

/// What this backend needs to redraw one actor: a mesh, and whichever of the two
/// depictions it produced last time.
type Drawable<'a> = (
    Actor<'a, CartoonStyle>,
    Option<&'a Mesh3d>,
    Option<&'a MeshMaterial3d<StandardMaterial>>,
);

/// Draws the ribbon as one of two different physical things.
///
/// **Opaque** is the default and the interesting one. The ribbon goes through
/// Bevy's ordinary opaque pass, so it is lit and it writes depth — and the
/// moment accumulation truncates every interval at that depth, which means a
/// volume in front of the ribbon dims it by exactly the absorbance along the
/// path between the eye and the surface. That is the composition the pathway
/// exists for: a structure inside the density map it was built from, with the
/// density in front correctly blocking the light from the ribbon behind, and no
/// sorting anywhere.
///
/// **Absorbing** makes the ribbon a medium of its own instead, which is worth
/// keeping for looking at a cartoon *through* another one.
///
/// An actor holds the components of exactly one mode; switching removes the
/// other's, or a ribbon would be drawn twice and deposit absorbance it no longer
/// claims to have.
pub fn draw_cartoons(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<Drawable>,
    layouts: Query<&CartoonLayout>,
) {
    for ((entity, style, colour, subset, bindings, dirty), mesh3d, material3d) in &dirty {
        if !dirty.any() {
            continue;
        }

        let flat = colour.flat.to_linear().to_f32_array();
        // Repaint in place when only the colouring moved. Worth the same as it
        // is for any lit mesh: rebuilding would re-solve every spline
        // to change four bytes a vertex.
        if !dirty.geometry
            && style.opaque
            && let Ok(layout) = layouts.get(entity)
            && let Some(colours) = cartoon::residue_colours(bindings, &store, &arrays, colour)
        {
            repaint(
                &mut meshes,
                mesh3d,
                cartoon::expand(&layout.residue, &colours, flat),
            );
        }

        if dirty.geometry
            && let Some((mesh, residue)) = build(bindings, subset, &store, &arrays, style, colour)
        {
            ensure_mesh(&mut commands, entity, &mut meshes, mesh3d, mesh);
            commands.entity(entity).insert(CartoonLayout { residue });
        }

        let mut actor = commands.entity(entity);
        if style.opaque {
            // Nothing to deposit: an opaque surface is where the integration
            // stops, not something it integrates through.
            actor.remove::<MomentVolume>();
            actor.remove::<MomentShell>();
            let material = StandardMaterial {
                // White when the mesh carries vertex colours, which multiply
                // into it; the flat colour otherwise.
                base_color: if bindings.get("colour").is_some() {
                    Color::WHITE
                } else {
                    colour.flat
                },
                perceptual_roughness: 0.85,
                // Both, and both are needed — `cull_mode` decides whether a back
                // face is drawn and `double_sided` whether it is lit as one.
                double_sided: true,
                cull_mode: None,
                ..default()
            };
            match material3d {
                Some(MeshMaterial3d(handle)) if materials.get(handle).is_some() => {
                    if let Some(mut slot) = materials.get_mut(handle) {
                        *slot = material;
                    }
                }
                _ => {
                    actor.insert(MeshMaterial3d(materials.add(material)));
                }
            }
            continue;
        }

        actor.remove::<MeshMaterial3d<StandardMaterial>>();
        actor.insert(MomentVolume {
            depiction: Depiction::Interior {
                sigma: style.sigma.max(0.0),
            },
            tint: transmission(colour),
        });
        match style.ior {
            Some(ior) => actor.insert(MomentShell {
                f0: normal_reflectance(ior),
                roughness: style.roughness.clamp(0.0, 1.0),
            }),
            None => actor.remove::<MomentShell>(),
        };
    }
}

/// Gives every placement of an opaque cartoon the mesh and material it holds.
///
/// [`place_volumes`](super::place_volumes) cannot do this: it queries for a
/// [`MomentVolume`], which an opaque actor deliberately does not have. Removal
/// is handled as well as insertion, so switching an actor to `absorbing` takes
/// the material off its copies rather than leaving them lit *and* absorbing.
#[allow(clippy::type_complexity)]
pub fn place_cartoons(
    mut commands: Commands,
    actors: Query<(&Mesh3d, Option<&MeshMaterial3d<StandardMaterial>>), With<CartoonStyle>>,
    placements: Query<(
        Entity,
        &Placement,
        Option<&Mesh3d>,
        Option<&MeshMaterial3d<StandardMaterial>>,
    )>,
) {
    for (entity, placement, mesh3d, material3d) in &placements {
        let Ok((mesh, material)) = actors.get(placement.0) else {
            // Not an opaque cartoon, or not drawn yet. Either way, nothing here.
            continue;
        };
        if mesh3d.map(|Mesh3d(handle)| handle.id()) != Some(mesh.0.id()) {
            commands.entity(entity).insert(mesh.clone());
        }
        let wanted = material.map(|MeshMaterial3d(handle)| handle.id());
        if material3d.map(|MeshMaterial3d(handle)| handle.id()) != wanted {
            match material {
                Some(material) => commands.entity(entity).insert(material.clone()),
                None => commands
                    .entity(entity)
                    .remove::<MeshMaterial3d<StandardMaterial>>(),
            };
        }
    }
}

/// Sweeps the ribbon and turns it into this pathway's mesh.
///
/// Normals are written for an opaque ribbon, which is lit and needs them, and
/// for an absorbing one only when a shell is on — exactly as
/// [`solid`](super::solid) does it. The accumulation itself cares where a
/// boundary is, not which way it faces, so a bare absorbing ribbon does not
/// carry twelve bytes a vertex for a pass that never reads them. The sweep
/// computes them either way; not writing them is the saving.
fn build(
    bindings: &Bindings,
    subset: &Subset,
    store: &DataStore,
    arrays: &Assets<DataArray>,
    style: &CartoonStyle,
    colour: &ColorBy,
) -> Option<(Mesh, Vec<u32>)> {
    let input = cartoon::read(bindings, subset, store, arrays)?;
    let ribbon = cartoon::build(&input.backbone(), &style.geometry);
    if ribbon.is_empty() {
        debug!("draw: a cartoon had no backbone to follow; nothing to draw");
        return None;
    }

    debug!(
        "draw: {} cartoon rebuilt — {} vertices, {} triangles",
        if style.opaque { "opaque" } else { "absorbing" },
        ribbon.positions.len(),
        ribbon.indices.len() / 3
    );

    // Only an opaque ribbon has anything to do with a colour array; see the
    // module docs.
    let colours = style.opaque.then(|| {
        let flat = colour.flat.to_linear().to_f32_array();
        match cartoon::residue_colours(bindings, store, arrays, colour) {
            Some(values) => cartoon::expand(&ribbon.residue, &values, flat),
            None => vec![flat; ribbon.positions.len()],
        }
    });
    let residue = ribbon.residue.clone();
    Some((
        mesh(ribbon, style.opaque || style.ior.is_some(), colours),
        residue,
    ))
}

fn mesh(ribbon: Ribbon, with_normals: bool, colours: Option<Vec<[f32; 4]>>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, ribbon.positions);
    if with_normals {
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, ribbon.normals);
    }
    if let Some(colours) = colours {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    }
    mesh.insert_indices(Indices::U32(ribbon.indices));
    mesh
}

/// Overwrites a mesh's vertex colours without touching anything else.
///
/// Four lines, written here rather than shared. Backends
/// duplicate their drawing code by design — see [`crate::draw`].
fn repaint(meshes: &mut Assets<Mesh>, existing: Option<&Mesh3d>, colours: Vec<[f32; 4]>) {
    let Some(Mesh3d(handle)) = existing else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(handle) else {
        return;
    };
    if mesh.count_vertices() != colours.len() {
        return;
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
}

/// Which residue each vertex of the built ribbon came from, so a colour change
/// repaints instead of re-solving every spline.
#[derive(Component, Debug)]
pub struct CartoonLayout {
    residue: Vec<u32>,
}

/// Replaces a mesh in place when the actor already has one.
///
/// The same four lines as [`solid`](super::solid)'s, written again rather
/// than shared. Backends duplicate their drawing code by design — see
/// [`crate::draw`] — and the early return is load-bearing: it ends the mutable
/// borrow of `meshes` before the second arm needs one.
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
