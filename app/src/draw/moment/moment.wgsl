// Accumulation pass: every fragment deposits its own signed contribution to the
// absorbance integral, using nothing but its own depth. That is what makes the
// result independent of the order the fragments arrive in.

#import bevy_render::view::View

struct Instance {
    world_from_local: mat4x4<f32>,
    tint: vec3<f32>,
    sigma: f32,
}

struct Bounds {
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var opaque_depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> bounds: Bounds;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // View-space distance in front of the camera, positive going away. The
    // moment domain is built from this, not from the clip-space depth: reverse-Z
    // is wildly non-uniform, and the antiderivative below is only the right one
    // for a linear depth.
    @location(0) view_z: f32,
    @location(1) @interpolate(flat) instance: u32,
}

struct MomentOutput {
    // b1..b4 of the scalar absorbance measure.
    @location(0) moments: vec4<f32>,
    // Per-channel total absorbance in rgb; b0 of the scalar measure in alpha.
    //
    // b0 has to be here because the moments are normalised by it before
    // reconstruction. It displaced the signed face count that used to occupy
    // this channel, which is a small loss: an unclosed mesh still shows up,
    // because a pairing that does not cancel drives b0 negative.
    @location(1) totals: vec4<f32>,
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
) -> MomentOutput {
    let instance = instances[in.instance];

    // Clamp rather than depth-test. Both endpoints of an interval get the same
    // clamp, so an interval entirely behind an occluder collapses to
    // F(z) - F(z) = 0 and one crossing the occluder is truncated at it. See the
    // note in pass.rs for why discarding the fragment instead is wrong.
    let depth = textureLoad(opaque_depth, vec2<i32>(floor(in.position.xy)), 0);
    let z = min(in.view_z, opaque_view_z(depth, in.position.xy));

    // Into the warped domain. Linear, so the antiderivatives below stay closed
    // form — §4 is explicit that a logarithmic warp would not.
    let span = max(bounds.far - bounds.near, 1e-6);
    let w = clamp((z - bounds.near) / span, 0.0, 1.0);

    // Front faces open an interval and back faces close it. Additive blending
    // performs the pairing, so no fragment has to find its partner — which is
    // exactly why nested and self-intersecting meshes need no special case.
    let orientation = select(1.0, -1.0, front_facing);

    // The measure has a density rather than a spike, so each moment has an
    // antiderivative and each fragment evaluates it at its own depth:
    //
    //   dA/dw = sigma * span   inside the mesh
    //   F_k(w) = sigma * span * w^(k+1) / (k+1)
    //
    // The `span` factor is the change of variable dz = span * dw. Dropping it
    // would make the absorbance depend on how wide the depth bound happened to
    // be this frame.
    let scale = instance.sigma * span * orientation;
    let w2 = w * w;

    var out: MomentOutput;
    out.moments = scale * vec4(
        w2 / 2.0,
        w2 * w / 3.0,
        w2 * w2 / 4.0,
        w2 * w2 * w / 5.0,
    );
    // k = 0 of the same family. Per channel for the colour, and once more
    // untinted for b0 — the moments above are of that same scalar measure, so
    // it is the right thing to normalise them by.
    out.totals = vec4(scale * w * (1.0 - instance.tint), scale * w);
    return out;
}
