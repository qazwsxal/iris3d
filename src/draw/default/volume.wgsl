// Accumulation pass for sampled grids: one fragment marches the whole ray and
// deposits the finished moments.
//
// # Why this is not the signed-prefix path
//
// `moment.wgsl` splits an interval across two fragments because a *mesh*
// fragment knows one depth and cannot find its partner. Front faces add
// `-F_k(w)`, back faces add `+F_k(w)`, and the additive blend performs the
// pairing.
//
// A grid has no such limitation. The box is closed, the field is a texture, and
// one fragment can walk the entire interval — so it integrates directly and the
// pairing never arises. That removes the closed-mesh requirement, and it removes
// the catastrophic cancellation a per-cell signed prefix would suffer: nothing
// here differences two antiderivatives extrapolated back to `w = 0`.
//
// The economy matters too. Rasterising every cell as a cube would be six faces
// of 16.7M cells for a 256^3 grid, at a depth complexity of ~512 blended
// fragments per pixel into two `Rgba32Float` targets. This is one fragment.
//
// # Why fullscreen rather than a box
//
// The obvious geometry is the grid's own bounding box, and it is a trap. With
// `cull_mode: None` a ray produces two fragments — front face and back face —
// and each would march the whole ray, so everything counts twice. Culling one
// facing fixes that but breaks the moment the camera enters the box: front faces
// are clipped away by the near plane, and back faces put the ray's origin behind
// its start.
//
// A fullscreen triangle has none of those cases. Exactly one fragment per pixel,
// inside the volume and out, and the slab test below rejects the pixels the box
// does not cover. What it costs is a slab test on pixels that miss, which is a
// handful of instructions against an entire class of geometry bugs.
//
// # What each step deposits
//
// The segment is treated as a thin slab of constant density — the density at
// its midpoint — and integrated with the *same* antiderivative the mesh path
// uses:
//
//   F_k(w) = sigma * span * w^(k+1) / (k+1)
//
// So a step contributes `sigma_mid * span * (w1^(k+1) - w0^(k+1)) / (k+1)`.
// That is the midpoint rule, second order in the step size, and it costs one
// texture tap plus the four powers already written for meshes.
//
// # Nearest neighbour versus linear is a sampler setting
//
// Because the density comes from one filtered tap, the reconstruction filter is
// not in this shader at all. `ImageFilterMode::Nearest` makes the field
// piecewise constant, which is exactly slab rendering per cell.
// `ImageFilterMode::Linear` makes it trilinear. Neither changes a line here.
//
// Exact per-cell integration of the trilinear form is a further step: on a ray
// a trilinear field is a *cubic*, so the segment integral has a closed form of
// degree `4 + k`. It is worth taking only if the midpoint rule proves visibly
// coarse, and it slots into the same loop.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View

struct Grid {
    // Maps the unit cube [0,1]^3 onto the grid's box in world space.
    world_from_local: mat4x4<f32>,
    // Its inverse, so the ray can be slab-tested in the space where the box is
    // axis-aligned and the local position doubles as the texture coordinate.
    local_from_world: mat4x4<f32>,
    // Linear RGB, and a *transmission* rather than a colour — the same meaning
    // `MomentVolume::tint` carries for a mesh. This is the medium's extinction
    // colour and is flat across the volume.
    //
    // Deliberately not the colour ramp. The ramp is what the volume *emits*, and
    // emission is `emit.wgsl`'s business; tinting the extinction with it as well
    // would count the same colour twice.
    tint: vec3<f32>,
    // Absorbance per world unit at a density of 1. The `opacity` control.
    sigma: f32,
    // Samples along the ray. A quality control: the step length divides out, so
    // the picture holds still as this moves.
    steps: f32,
    // How much light the volume gives off per unit of path length at a strength
    // of 1. Unused here; `emit.wgsl` reads it from the same uniform so the two
    // passes cannot disagree about the volume they are drawing.
    emission: f32,
    _pad: vec2<u32>,
}

// Group 0 is the accumulation pass's own layout, shared with `moment.wgsl`.
// Binding 1 there is the mesh instance buffer, which this shader has no use for
// — a bind group entry a shader does not declare is legal and costs nothing, so
// the layout is reused rather than duplicated.
@group(0) @binding(0) var<uniform> view: View;
#ifdef MULTISAMPLED
@group(0) @binding(2) var opaque_depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(2) var opaque_depth: texture_depth_2d;
#endif
@group(0) @binding(3) var<uniform> bounds: Bounds;

struct Bounds {
    near: f32,
    far: f32,
}

// The grid's own data. A separate group because a texture cannot live in the
// shared instance buffer, so each grid is its own draw with its own binding.
@group(1) @binding(0) var<uniform> grid: Grid;
// What absorbs in `r`, what the ramp is read by in `g`, what emits in `b`.
// Separate choices but not separate fetches. Only `r` is read here — this pass
// deposits absorbance and nothing else; `g` and `b` are `emit.wgsl`'s.
@group(1) @binding(1) var field: texture_3d<f32>;
@group(1) @binding(2) var field_sampler: sampler;

struct MomentOutput {
    @location(0) moments: vec4<f32>,
    @location(1) totals: vec4<f32>,
}

// The ray through a pixel, in world space, with `t = 0` at the camera.
//
// Both projections are handled in view space, where the difference between them
// is one branch rather than two code paths: a perspective ray leaves the origin,
// an orthographic one leaves the near plane travelling down -Z. `world_from_view`
// is rigid, so `t` means the same distance in both spaces and the slab test
// below can be trusted with it.
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    // View-space depth at `t = 0`, and how it grows per unit of `t`. Affine, so
    // two numbers describe the whole ray and the moment domain never needs a
    // matrix again.
    z_at_origin: f32,
    z_per_t: f32,
}

fn ray_through(uv: vec2<f32>) -> Ray {
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reverse-Z, so 1.0 is the near plane — the one depth that is finite under
    // an infinite far plane.
    var near = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    near = near / near.w;

    // 1 for an orthographic projection, 0 for a perspective one.
    let orthographic = view.clip_from_view[3][3] > 0.5;
    var origin_view: vec3<f32>;
    var direction_view: vec3<f32>;
    if orthographic {
        origin_view = vec3(near.xy, 0.0);
        direction_view = vec3(0.0, 0.0, -1.0);
    } else {
        origin_view = vec3(0.0);
        direction_view = normalize(near.xyz);
    }

    var ray: Ray;
    ray.origin = (view.world_from_view * vec4(origin_view, 1.0)).xyz;
    ray.direction = (view.world_from_view * vec4(direction_view, 0.0)).xyz;
    // View-space depth is -z, and both are affine in t.
    ray.z_at_origin = -origin_view.z;
    ray.z_per_t = -direction_view.z;
    return ray;
}

// Turns a sampled depth value back into a view-space distance. The same
// arithmetic `moment.wgsl` uses, and unprojected rather than closed form for the
// same reason: an orthographic camera has to work too.
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

// Where the ray enters and leaves the unit cube, as a range of `t`.
//
// The slab test runs in local space, where the box is axis-aligned, but `t` is
// shared with world space: an affine transform carries the ray parameter
// unchanged, so a distance found here is a distance there.
fn box_span(origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    // A component of exactly zero would divide to an infinity of the wrong sign
    // on one side of the slab. Nudging it keeps both bounds finite and the
    // min/max below sorts them out.
    let safe = select(direction, vec3(1e-20), abs(direction) < vec3(1e-20));
    let first = (vec3(0.0) - origin) / safe;
    let second = (vec3(1.0) - origin) / safe;
    let low = min(first, second);
    let high = max(first, second);
    return vec2(max(max(low.x, low.y), low.z), min(min(high.x, high.y), high.z));
}

// The four moments of one step, with `F_k(w) = scale * w^(k+1) / (k+1)`.
//
// `emit.wgsl` repeats this arithmetic to subtract its own contribution from the
// buffer, and that subtraction is only exact while the two agree to the last
// bit. Change one and change the other.
fn step_moments(scale: f32, w0: f32, w1: f32) -> vec4<f32> {
    let a0 = w0 * w0;
    let a1 = w1 * w1;
    return scale * vec4(
        (a1 - a0) / 2.0,
        (a1 * w1 - a0 * w0) / 3.0,
        (a1 * a1 - a0 * a0) / 4.0,
        (a1 * a1 * w1 - a0 * a0 * w0) / 5.0,
    );
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> MomentOutput {
    var out: MomentOutput;
    out.moments = vec4(0.0);
    out.totals = vec4(0.0);

    let ray = ray_through(in.uv);
    let local_origin = (grid.local_from_world * vec4(ray.origin, 1.0)).xyz;
    // A direction transforms with w = 0, and is deliberately *not* renormalised:
    // that would break the shared parameterisation the slab test relies on.
    let local_direction = (grid.local_from_world * vec4(ray.direction, 0.0)).xyz;

    let span_t = box_span(local_origin, local_direction);
    // Behind the camera is clipped here rather than by geometry, which is what
    // lets the camera sit inside the volume with no special case.
    var start = max(span_t.x, 0.0);
    var end = span_t.y;
    if end <= start {
        return out;
    }

    // Clamp to the occluder rather than depth-testing, for the same reason the
    // mesh path does: what lies behind an opaque surface must contribute
    // nothing, and truncating the interval is what makes the resolve's total
    // exactly the absorbance in front of that surface.
    //
    // Sample 0 under MSAA. Reading `@builtin(sample_index)` would run this whole
    // march once per sample to refine the one place they differ.
    let coord = vec2<i32>(floor(in.position.xy));
    let occluder_z = opaque_view_z(textureLoad(opaque_depth, coord, 0), in.position.xy);
    if ray.z_per_t > 1e-6 {
        end = min(end, (occluder_z - ray.z_at_origin) / ray.z_per_t);
    }
    if end <= start {
        return out;
    }

    let steps = max(i32(grid.steps), 1);
    let step_t = (end - start) / f32(steps);
    let domain = max(bounds.far - bounds.near, 1e-6);

    var moments = vec4(0.0);
    var scalar = 0.0;

    for (var step = 0; step < steps; step = step + 1) {
        let t0 = start + f32(step) * step_t;
        let t1 = t0 + step_t;

        // The midpoint, which is where the slab's constant density is read.
        let mid = local_origin + (t0 + t1) * 0.5 * local_direction;
        let density = textureSampleLevel(field, field_sampler, mid, 0.0).r;
        // Empty space contributes to no moment, so skipping it is exact rather
        // than an approximation. It is also the only empty-space skipping here.
        if density <= 0.0 {
            continue;
        }

        // Into the warped domain, which is linear — so the antiderivative is the
        // one the mesh path uses, unchanged.
        let w0 = clamp((ray.z_at_origin + t0 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);
        let w1 = clamp((ray.z_at_origin + t1 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);

        // The `domain` factor is the change of variable dz = span * dw. Dropping
        // it would make the absorbance depend on how wide the depth bound
        // happened to be this frame.
        let scale = density * grid.sigma * domain;
        moments = moments + step_moments(scale, w0, w1);
        scalar = scalar + scale * (w1 - w0);
    }

    out.moments = moments;
    // k = 0 of the same family: per channel for the colour, and once more
    // untinted for b0, which is what the moments are normalised by.
    out.totals = vec4(scalar * (1.0 - grid.tint), scalar);
    return out;
}
