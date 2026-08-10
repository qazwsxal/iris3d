// Accumulation pass: every fragment deposits its own signed contribution to the
// absorbance integral, using nothing but its own depth. That is what makes the
// result independent of the order the fragments arrive in.

#import bevy_render::view::View

struct Instance {
    world_from_local: mat4x4<f32>,
    tint: vec3<f32>,
    sigma: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var opaque_depth: texture_depth_2d;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // View-space distance in front of the camera, positive going away. The
    // moment domain is this, not the clip-space depth: reverse-Z is wildly
    // non-uniform, and the antiderivative below is only the right one for a
    // linear depth.
    @location(0) view_z: f32,
    @location(1) @interpolate(flat) instance: u32,
}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) local_position: vec3<f32>,
) -> VertexOutput {
    let instance = instances[instance_index];
    let world = instance.world_from_local * vec4(local_position, 1.0);

    var out: VertexOutput;
    out.position = view.clip_from_world * world;
    out.view_z = -(view.view_from_world * world).z;
    out.instance = instance_index;
    return out;
}

// Turns a sampled depth value back into a view-space distance.
//
// Unprojected through `view_from_clip` rather than with the closed form for an
// infinite reverse-Z perspective, so that an orthographic camera works too.
fn opaque_view_z(depth: f32, fragment_xy: vec2<f32>) -> f32 {
    // Reverse-Z clears to zero, so zero means nothing was drawn here — the
    // occluder is infinitely far away and clamps nothing.
    if depth <= 0.0 {
        return 3.4e38;
    }
    let uv = (fragment_xy - view.viewport.xy) / view.viewport.zw;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let unprojected = view.view_from_clip * vec4(ndc, depth, 1.0);
    return -unprojected.z / unprojected.w;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    let instance = instances[in.instance];

    // Clamp rather than depth-test. Both endpoints of an interval get the same
    // clamp, so an interval entirely behind an occluder collapses to
    // F(z) - F(z) = 0 and one crossing the occluder is truncated at it. See the
    // note in pass.rs for why discarding the fragment instead is wrong.
    let depth = textureLoad(opaque_depth, vec2<i32>(floor(in.position.xy)), 0);
    let z = min(in.view_z, opaque_view_z(depth, in.position.xy));

    // Front faces open an interval and back faces close it. Additive blending
    // performs the pairing, so no fragment has to find its partner — which is
    // exactly why nested and self-intersecting meshes need no special case.
    let orientation = select(1.0, -1.0, front_facing);

    // F(z) = sigma * z, the antiderivative of a uniform interior density.
    // Per channel, so a red volume in front of a blue one composes correctly:
    // `tint` is what the medium transmits, so `1 - tint` is what it takes.
    // Geometry behind the camera is clamped away rather than counted backwards.
    let absorbance = instance.sigma * max(z, 0.0) * (1.0 - instance.tint);

    // Alpha carries the signed face count. A closed mesh cancels it to zero,
    // so a pixel where it does not is a hole in the geometry.
    return vec4(orientation * absorbance, orientation);
}
