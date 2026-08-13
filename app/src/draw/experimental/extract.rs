//! Getting the volumes across to the render world.
//!
//! One flat list per frame rather than an entity per volume in the render
//! world. The moment pass draws every volume in one go with no sorting, no
//! batching and no per-entity state, so a list is all the pass can use — and
//! rebuilding it costs a matrix copy per volume, against a scene that is only
//! redrawn when something changes.
//!
//! What is extracted is the *placement*, not the actor. An actor owns the mesh
//! and is permanently hidden (see [`crate::scene::link`]), so the inherited
//! visibility filter below is what keeps the actor's own copy out of the list
//! while admitting every placement of it.

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::render::Extract;
use bevy::render::render_resource::ShaderType;

use super::{MomentShell, MomentVolume};

/// One absorbing volume, ready to draw.
pub struct ExtractedVolume {
    pub mesh: AssetId<Mesh>,
    pub world_from_local: Mat4,
    /// The inverse transpose of the world matrix's rotation and scale, which is
    /// what a normal transforms by. Only the shell needs it — the accumulation
    /// cares where a face is, not which way it points — but it is computed here
    /// rather than in the shader because it is per volume, not per vertex.
    pub world_from_local_normal: Mat3,
    /// Strength and mode, already packed for the shader.
    pub strength: f32,
    pub dirac: u32,
    pub tint: Vec3,
    /// The dielectric skin over this volume, if it was asked for.
    pub shell: Option<MomentShell>,
    /// The eight world-space corners of the volume's bounds.
    ///
    /// Carried so the moment domain can be fitted to what is actually on
    /// screen. Corners rather than a centre and extent because the bound is
    /// wanted along the view axis, and a rotated box's extent along an
    /// arbitrary axis is not recoverable from its local one.
    pub corners: [Vec3; 8],
}

#[derive(Resource, Default)]
pub struct ExtractedVolumes(pub Vec<ExtractedVolume>);

/// One sampled grid, ready to draw.
///
/// Kept in its own list rather than folded into [`ExtractedVolume`] because the
/// two are drawn by different pipelines with different bind groups: a mesh is
/// geometry with a shared instance buffer, a grid is a fullscreen pass with a
/// texture of its own. Sharing a list would mean every consumer branching on
/// which kind it had.
pub struct ExtractedGrid {
    /// Maps the unit cube onto the grid's box in world space, and its inverse.
    /// Both are carried rather than inverted in the shader: it is one inversion
    /// per grid per frame here against one per pixel there.
    pub world_from_local: Mat4,
    pub local_from_world: Mat4,
    pub tint: Vec3,
    pub sigma: f32,
    pub steps: f32,
    pub emission: f32,
    pub field: Handle<Image>,
    pub ramp: Handle<Image>,
    /// The eight world-space corners of the box, for fitting the moment domain.
    /// Taken from the grid rather than an `Aabb`, which is exact and available
    /// on the first frame — a grid states its own extent, so unlike a mesh it
    /// never has to wait to be measured.
    pub corners: [Vec3; 8],
}

#[derive(Resource, Default)]
pub struct ExtractedGrids(pub Vec<ExtractedGrid>);

/// Copies every visible sampled grid into the render world.
///
/// Handles rather than `AssetId`s, because the render world resolves them
/// through `RenderAssets<GpuImage>` and a bare id would not keep the asset
/// alive against a scene that drops the actor mid-frame.
#[allow(clippy::type_complexity)]
pub fn extract_grids(
    mut commands: Commands,
    grids: Extract<
        Query<(
            &super::volume::GridField,
            &super::volume::GridStyle,
            &super::volume::GridBox,
            &GlobalTransform,
            &InheritedVisibility,
        )>,
    >,
) {
    let extracted = grids
        .iter()
        .filter(|(_, _, _, _, visibility)| visibility.get())
        .map(|(field, style, grid, transform, _)| {
            let world_from_local = transform.to_matrix() * grid.cube_to_box();
            ExtractedGrid {
                world_from_local,
                local_from_world: world_from_local.inverse(),
                tint: field.tint,
                sigma: style.sigma,
                steps: style.steps,
                emission: style.emission,
                field: field.field.clone(),
                ramp: field.ramp.clone(),
                corners: cube_corners(world_from_local),
            }
        })
        .collect();

    commands.insert_resource(ExtractedGrids(extracted));
}

/// The world-space corners of the unit cube under a transform.
fn cube_corners(world_from_local: Mat4) -> [Vec3; 8] {
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let pick = |bit: usize| if index & (1 << bit) == 0 { 0.0 } else { 1.0 };
        *corner =
            world_from_local.transform_point3(Vec3::new(pick(0), pick(1), pick(2)));
    }
    corners
}

/// Copies every visible absorbing volume into the render world.
///
/// Visibility is the inherited flag rather than the per-view one. The list is
/// shared by all views, so per-view frustum culling has nowhere to go here yet;
/// it belongs with the phase item this pass will eventually grow.
#[allow(clippy::type_complexity)]
pub fn extract_volumes(
    mut commands: Commands,
    volumes: Extract<
        Query<(
            &MomentVolume,
            &Mesh3d,
            &GlobalTransform,
            &InheritedVisibility,
            Option<&Aabb>,
            Option<&MomentShell>,
        )>,
    >,
) {
    let extracted = volumes
        .iter()
        .filter(|(_, _, _, visibility, _, _)| visibility.get())
        .map(|(volume, mesh, transform, _, aabb, shell)| {
            let world_from_local = transform.to_matrix();
            let (strength, dirac) = volume.depiction.packed();
            ExtractedVolume {
                mesh: mesh.0.id(),
                world_from_local,
                world_from_local_normal: Mat3::from_mat4(world_from_local)
                    .inverse()
                    .transpose(),
                strength,
                dirac,
                tint: volume.tint,
                shell: shell.copied(),
                corners: corners_of(aabb, world_from_local),
            }
        })
        .collect();

    commands.insert_resource(ExtractedVolumes(extracted));
}

/// How many lights the shell reflects. Beyond this they are ignored, in the
/// order the query happens to return them.
///
/// Four because a specular highlight is a small bright spot and a scene with
/// more key lights than that is not one iris3d sets up: the viewport spawns a
/// key and a fill. This is a fixed-size uniform rather than a storage buffer
/// because the count is small and known.
pub const MAX_SHELL_LIGHTS: usize = 4;

/// One directional light, as the shell shader wants it.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct ShellLight {
    /// Unit vector pointing *towards* the light, which is what a half vector
    /// wants. A `DirectionalLight` faces along its own -Z.
    pub towards: Vec3,
    /// Illuminance, scaled down to something a specular term can use directly.
    pub intensity: f32,
    pub colour: Vec3,
    pub _pad: f32,
}

/// The lights the shell reflects, and what it reflects when no light is in that
/// direction.
#[derive(Clone, Default, ShaderType)]
pub struct ShellLighting {
    pub lights: [ShellLight; MAX_SHELL_LIGHTS],
    /// The viewport's own clear colour, in linear RGB.
    ///
    /// What a glass object is surrounded by *is* the background, so that is
    /// what it should mirror. Taken from the 3D camera rather than invented, so
    /// changing the viewport's background changes the reflections with it.
    ///
    /// One colour for every view, because iris3d draws its 3D scene through a
    /// single camera. A second 3D camera with a different background would
    /// reflect the first one's, which is worth knowing before adding one.
    pub background: Vec3,
    pub count: u32,
}

#[derive(Resource, Default)]
pub struct ExtractedShellLighting(pub ShellLighting);

/// Copies the scene's directional lights across for the shell to reflect.
///
/// The shell is the only lit thing this pathway draws — an absorbing volume has
/// no surface to shade — so none of Bevy's own light bindings are set up for
/// these views, and there is nothing to reuse. Four directions and four colours
/// is the whole of what a specular term needs, so they are extracted directly
/// rather than by standing up the full clustered-lighting machinery for one
/// pass.
///
/// Illuminance is divided down because it is quoted in lux, which is the right
/// unit for a physically-based diffuse response and a very large number for a
/// highlight that is simply added to the target.
pub fn extract_shell_lighting(
    mut commands: Commands,
    lights: Extract<Query<(&DirectionalLight, &GlobalTransform, &InheritedVisibility)>>,
    cameras: Extract<Query<&Camera, With<Camera3d>>>,
    clear_colour: Extract<Res<ClearColor>>,
) {
    /// Lux to something that reads as a highlight rather than a white blob.
    const SCALE: f32 = 1.0 / 6000.0;

    let mut lighting = ShellLighting::default();
    for (light, transform, visibility) in lights.iter() {
        if !visibility.get() || lighting.count as usize >= MAX_SHELL_LIGHTS {
            continue;
        }
        lighting.lights[lighting.count as usize] = ShellLight {
            towards: -transform.forward().as_vec3(),
            intensity: light.illuminance * SCALE,
            colour: LinearRgba::from(light.color).to_vec3(),
            _pad: 0.0,
        };
        lighting.count += 1;
    }

    // Whatever the 3D view clears to is what surrounds the object, so it is
    // what the object reflects. `None` means the camera clears nothing and
    // draws over whatever was there, which no colour describes — black is the
    // honest answer rather than borrowing the global one.
    let background = cameras
        .iter()
        .next()
        .map(|camera| match camera.clear_color {
            ClearColorConfig::Custom(colour) => LinearRgba::from(colour).to_vec3(),
            ClearColorConfig::Default => LinearRgba::from(clear_colour.0).to_vec3(),
            ClearColorConfig::None => Vec3::ZERO,
        })
        .unwrap_or(Vec3::ZERO);
    lighting.background = background;

    commands.insert_resource(ExtractedShellLighting(lighting));
}

/// The world-space corners of a volume's local bounds.
///
/// A missing `Aabb` means the mesh has not been measured yet, which happens on
/// the frame it appears. A degenerate box at the origin of the volume is the
/// safe answer: it contributes nothing to the depth bound, and the next frame
/// has the real one.
fn corners_of(aabb: Option<&Aabb>, world_from_local: Mat4) -> [Vec3; 8] {
    let Some(aabb) = aabb else {
        return [world_from_local.transform_point3(Vec3::ZERO); 8];
    };
    let (low, high) = (Vec3::from(aabb.min()), Vec3::from(aabb.max()));
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let pick = |bit: usize, low: f32, high: f32| {
            if index & (1 << bit) == 0 { low } else { high }
        };
        *corner = world_from_local.transform_point3(Vec3::new(
            pick(0, low.x, high.x),
            pick(1, low.y, high.y),
            pick(2, low.z, high.z),
        ));
    }
    corners
}
