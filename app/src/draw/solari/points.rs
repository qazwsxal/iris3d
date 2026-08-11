//! Point clouds, drawn as instanced spheres.
//!
//! The default backend draws camera-facing quads, which cannot be raytraced:
//! they are a vertex-shader trick and a ray knows nothing about the camera that
//! spawned it. A raytraced point has to be real geometry, so it is a sphere.
//!
//! One sphere asset, one instance per point. Merging instead would be twenty
//! million triangles for a quarter-million points; instancing is one sphere's
//! worth of geometry and a transform each. Colour comes from the material
//! palette rather than from vertices, because every instance shares the one
//! mesh — see [`super::RampPalette`].

use bevy::prelude::*;

use crate::scene::registry::{ActorKind, ActorRegistry, ParamKind, ParamSpec, float};
use crate::scene::{DataArray, DataStore, Dtype};

use super::{
    Actor, Dirty, FlatMaterial, Instance, Instances, Primitives, RampPalette, bound, ensure_flat,
    mark, normalised,
};

/// Points drawn as spheres. `size` is a diameter in world units, so a sensible
/// value depends on the data's own scale.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct PointSpheresStyle {
    pub size: f32,
}

const PARAMS: &[ParamSpec] = &[
    ParamSpec {
        id: "positions",
        label: "positions",
        kind: ParamKind::Array {
            dtypes: &[Dtype::Float32],
            shape: &[0, 3],
            required: true,
        },
    },
    ParamSpec {
        id: "colour",
        label: "colour by",
        kind: ParamKind::Array {
            dtypes: &[],
            shape: &[0],
            required: false,
        },
    },
    ParamSpec {
        id: "size",
        label: "size",
        kind: ParamKind::Float {
            default: 0.05,
            min: 0.001,
            max: 1.0,
            logarithmic: true,
        },
    },
];

pub fn register(registry: &mut ActorRegistry) {
    registry.register(ActorKind {
        id: "points",
        label: "points",
        // A point cloud is a point cloud whichever pathway draws it. Spheres
        // rather than billboards is this backend's answer, not part of what the
        // id means.
        shared: true,
        params: PARAMS,
        apply: |entity, params| {
            entity.insert(PointSpheresStyle {
                size: float(params, "size", 0.05),
            });
        },
    });
}

/// `size` is baked into each instance's transform, so changing it moves
/// geometry rather than rewriting a uniform.
pub fn invalidate(mut commands: Commands, changed: Query<Entity, Changed<PointSpheresStyle>>) {
    for entity in &changed {
        mark(&mut commands, entity, Dirty::GEOMETRY);
    }
}

pub fn draw_points(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut palette: ResMut<RampPalette>,
    primitives: Res<Primitives>,
    arrays: Res<Assets<DataArray>>,
    store: Res<DataStore>,
    dirty: Query<(Actor<PointSpheresStyle>, Option<&FlatMaterial>)>,
) {
    for ((entity, style, colour, subset, bindings, dirty), existing) in &dirty {
        if !dirty.any() {
            continue;
        }
        let Some(array) = bound(bindings, "positions", &store, &arrays) else {
            continue;
        };

        let all = array.to_vec3();
        if all.is_empty() {
            continue;
        }
        // Points have no connectivity, so a subset is a plain filter: nothing
        // refers to a point by index, so nothing needs renumbering.
        let kept = subset.selected(all.len(), &arrays);

        // Computed over the whole field and then narrowed, so a subset does not
        // shift where a value lands in the colour map.
        let ramp = bound(bindings, "colour", &store, &arrays)
            .and_then(|values| normalised(values, colour, array.count() as usize));

        // A diameter, and the sphere has radius 1.
        let scale = Vec3::splat(style.size * 0.5);
        let placed = |index: usize| Transform::from_translation(all[index]).with_scale(scale);
        let indices: Vec<usize> = match &kept {
            Some(kept) => kept.iter().map(|index| *index as usize).collect(),
            None => (0..all.len()).collect(),
        };

        // Two branches rather than one with a per-point conditional: with
        // nothing bound the whole cloud shares one material, and looking that
        // up per point would return the same handle a quarter of a million
        // times.
        let sphere = || primitives.sphere.clone();
        let items: Vec<Instance> = match &ramp {
            Some(ramp) => indices
                .into_iter()
                .map(|index| Instance {
                    mesh: sphere(),
                    transform: placed(index),
                    material: palette.pick(colour.map, &mut materials, ramp[index]),
                })
                .collect(),
            None => {
                let flat = ensure_flat(
                    &mut commands,
                    entity,
                    &mut materials,
                    existing,
                    colour.flat.to_linear(),
                );
                indices
                    .into_iter()
                    .map(|index| Instance {
                        mesh: sphere(),
                        transform: placed(index),
                        material: flat.clone(),
                    })
                    .collect()
            }
        };

        debug!("draw: solari, {} point spheres", items.len());
        commands.entity(entity).insert(Instances(items));
    }
}
