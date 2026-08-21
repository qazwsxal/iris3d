// Emission pass for sampled grids: what the volume *adds*, attenuated by
// everything in front of it.
//
// A scientific volume emits as well as absorbs. `volume.wgsl` deposits the
// absorbing half into the shared moment buffer; this pass marches the same ray
// again and integrates
//
//   L = integral of  j(s) * T(0 -> s)  ds
//
// where `j` is emission per unit path length and `T(0 -> s)` is the
// transmittance from the eye to `s`.
//
// # Why this cannot be folded into the first pass
//
// `T(0 -> s)` must account for *everything* in front of `s`, not only this
// volume: another volume overlapping it, an absorbing mesh passing through it, a
// piece of glass in front of it. A single-pass march knows its own absorbance
// and nothing else, so it would light an occluded volume as though it were
// unoccluded.
//
// Running after the accumulation solves that, because by then the moment buffer
// holds every absorber on the ray. This is the same structure the shell uses —
// see `shell.wgsl` — except that a shell queries at one depth and a volume
// queries at every step.
//
// # The subtraction, which is what makes the common case exact
//
// The reconstruction in `reconstruct.wgsl` is a lower bound, and it is loose
// exactly where a volume lives: a continuous measure queried in its own interior
// comes back 0.26 against a true 0.50. Applied naively, a volume would light
// itself at roughly twice the brightness it should.
//
// Moments are additive, so that is avoidable. This pass recomputes its own
// contribution — the same numbers `volume.wgsl` deposited, from the same field
// and the same ray — and subtracts it from the buffer:
//
//   b_others = b_global - b_self          exact, component by component
//   A_front(z) = A_self(z) + A_others(z)
//
// `A_self(z)` is accumulated exactly while marching front to back, so the loose
// reconstruction is applied *only* to what other objects contributed. One volume
// with nothing overlapping it leaves `b_others` at zero and is attenuated
// exactly. The bound's error survives only where things genuinely overlap, which
// is the part no bookkeeping recovers.
//
// The cost is one extra traversal: the ray is marched once to total up `b_self`,
// then again to emit. The first loop is cheap — one tap per step and no
// reconstruction — so this is well short of twice the work.
//
// # Where it runs
//
// After the resolve, blending additively. The resolve has already dimmed what
// lies behind the volume by `exp(-A)`; this is the light the volume puts back
// on top, and it must not be dimmed again by the same absorbance it already
// accounted for internally.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::view::View
#import iris3d::moment_reconstruct::absorbed_fraction

struct Grid {
    world_from_local: mat4x4<f32>,
    local_from_world: mat4x4<f32>,
    tint: vec3<f32>,
    sigma: f32,
    steps: f32,
    // How much light the volume gives off per unit of path length at a strength
    // of 1. Separate from `sigma` so a volume can be made brighter without
    // becoming more opaque, which is the pairing scientific volume rendering
    // usually wants.
    emission: f32,
    _pad: vec2<u32>,
}

struct Bounds {
    near: f32,
    far: f32,
}

// Group 0 repeats the resolve pass's binding order exactly — view, moments,
// totals, depth, bound — so the two read the same numbers from the same places
// and neither drifts when the moment target changes shape. It is a separate
// layout only because this shader also runs a vertex stage.
@group(0) @binding(0) var<uniform> view: View;
#ifdef MULTISAMPLED
@group(0) @binding(1) var moments_texture: texture_multisampled_2d<f32>;
@group(0) @binding(2) var totals_texture: texture_multisampled_2d<f32>;
@group(0) @binding(3) var opaque_depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var moments_texture: texture_2d<f32>;
@group(0) @binding(2) var totals_texture: texture_2d<f32>;
@group(0) @binding(3) var opaque_depth: texture_depth_2d;
#endif
@group(0) @binding(4) var<uniform> bounds: Bounds;

@group(1) @binding(0) var<uniform> grid: Grid;
@group(1) @binding(1) var field: texture_3d<f32>;
@group(1) @binding(2) var field_sampler: sampler;
// The colour map as a 1D image. Here it is what the volume *emits*, so it is an
// ordinary colour rather than the transmission `tint` is.
@group(1) @binding(3) var ramp: texture_2d<f32>;
@group(1) @binding(4) var ramp_sampler: sampler;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    z_at_origin: f32,
    z_per_t: f32,
}

fn ray_through(uv: vec2<f32>) -> Ray {
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    var near = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    near = near / near.w;

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
    ray.z_at_origin = -origin_view.z;
    ray.z_per_t = -direction_view.z;
    return ray;
}

fn opaque_view_z(depth: f32, fragment_xy: vec2<f32>) -> f32 {
    if depth <= 0.0 {
        return 3.4e38;
    }
    let uv = (fragment_xy - view.viewport.xy) / view.viewport.zw;
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let unprojected = view.view_from_clip * vec4(ndc, depth, 1.0);
    return -unprojected.z / unprojected.w;
}

fn box_span(origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    let safe = select(direction, vec3(1e-20), abs(direction) < vec3(1e-20));
    let first = (vec3(0.0) - origin) / safe;
    let second = (vec3(1.0) - origin) / safe;
    let low = min(first, second);
    let high = max(first, second);
    return vec2(max(max(low.x, low.y), low.z), min(min(high.x, high.y), high.z));
}

// Must stay bit-for-bit identical to `volume.wgsl`'s. The subtraction below is
// only exact while the two agree.
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
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let nothing = vec4(0.0, 0.0, 0.0, 1.0);

    let ray = ray_through(in.uv);
    let local_origin = (grid.local_from_world * vec4(ray.origin, 1.0)).xyz;
    let local_direction = (grid.local_from_world * vec4(ray.direction, 0.0)).xyz;

    let span_t = box_span(local_origin, local_direction);
    var start = max(span_t.x, 0.0);
    var end = span_t.y;
    if end <= start {
        return nothing;
    }

    let coord = vec2<i32>(floor(in.position.xy));
    let occluder_z = opaque_view_z(textureLoad(opaque_depth, coord, 0), in.position.xy);
    if ray.z_per_t > 1e-6 {
        end = min(end, (occluder_z - ray.z_at_origin) / ray.z_per_t);
    }
    if end <= start {
        return nothing;
    }

    let steps = max(i32(grid.steps), 1);
    let step_t = (end - start) / f32(steps);
    let domain = max(bounds.far - bounds.near, 1e-6);

    // First traversal: total up what *this* volume put into the buffer. No
    // reconstruction and no ramp lookup, so it is one tap and a few multiplies
    // per step.
    var self_moments = vec4(0.0);
    var self_scalar = 0.0;
    for (var step = 0; step < steps; step = step + 1) {
        let t0 = start + f32(step) * step_t;
        let t1 = t0 + step_t;
        let mid = local_origin + (t0 + t1) * 0.5 * local_direction;
        let density = textureSampleLevel(field, field_sampler, mid, 0.0).r;
        if density <= 0.0 {
            continue;
        }
        let w0 = clamp((ray.z_at_origin + t0 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);
        let w1 = clamp((ray.z_at_origin + t1 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);
        let scale = density * grid.sigma * domain;
        self_moments = self_moments + step_moments(scale, w0, w1);
        self_scalar = self_scalar + scale * (w1 - w0);
    }

    // What everybody else deposited. Exact, because the accumulation this is
    // subtracted from is additive.
    //
    // Cancellation here is self-limiting rather than dangerous: when this volume
    // dominates the pixel the difference is noisy, but it is also near zero, so
    // it multiplies into nothing below.
    let global_moments = textureLoad(moments_texture, coord, 0);
    let global_totals = textureLoad(totals_texture, coord, 0);
    let others_moments = global_moments - self_moments;
    let others_scalar = global_totals.a - self_scalar;
    // Per channel, because extinction is coloured. The depth *distribution* is
    // taken to be common across channels, which is the assumption the
    // two-attachment split already rests on and which the shell also makes.
    let others_tinted = max(global_totals.rgb - self_scalar * (1.0 - grid.tint), vec3(0.0));
    let others_present = others_scalar > 1e-5;

    // Second traversal: emit.
    var radiance = vec3(0.0);
    // This volume's own absorbance in front of the current step, per channel.
    // Accumulated rather than reconstructed, so it carries no bound error.
    var self_front = vec3(0.0);

    for (var step = 0; step < steps; step = step + 1) {
        let t0 = start + f32(step) * step_t;
        let t1 = t0 + step_t;
        let mid = local_origin + (t0 + t1) * 0.5 * local_direction;
        // r absorbs, g is read by the ramp, b emits. Three independent
        // quantities in one fetch — see `field_texture` in volume.rs, which also
        // does the falling back, so nothing here has to ask which is bound.
        let sampled = textureSampleLevel(field, field_sampler, mid, 0.0).rgb;
        let density = sampled.r;
        let glow = sampled.b;
        // Both, because they are now different fields: a step that absorbs
        // nothing may still emit, and a step that emits nothing may still
        // shadow what is behind it. Testing only the density would drop the
        // first, which is exactly the case separate arrays exist to express.
        if density <= 0.0 && glow <= 0.0 {
            continue;
        }

        let w0 = clamp((ray.z_at_origin + t0 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);
        let w1 = clamp((ray.z_at_origin + t1 * ray.z_per_t - bounds.near) / domain, 0.0, 1.0);
        // World path length of this step, which is what an emission per unit
        // length has to be multiplied by.
        let path = (w1 - w0) * domain;

        // Everything else in front of this step. The reconstruction runs on the
        // others' moments alone, which is the whole point of the subtraction:
        // with nothing else on the ray this term is exactly zero and the
        // attenuation reduces to the accumulated `self_front`.
        var others_front = vec3(0.0);
        if others_present {
            let fraction = absorbed_fraction(others_moments / others_scalar, w0);
            others_front = others_tinted * fraction;
        }

        // Attenuated at the step's *entry*, which is where `self_front` stands.
        // A slab does not shadow its own emission; its absorbance applies to
        // what follows it.
        let transmittance = exp(-(self_front + others_front));

        let colour = textureSampleLevel(ramp, ramp_sampler, vec2(sampled.g, 0.5), 0.0).rgb;
        // Emission is driven by `glow`, not by the density. That separation is
        // the whole point of the third input: a cloud's density decides what it
        // blocks and its temperature decides what it radiates.
        radiance = radiance + colour * grid.emission * glow * path * transmittance;

        // Carry this step's own absorbance forward, tinted exactly as the
        // accumulation tinted it.
        self_front = self_front + density * grid.sigma * path * (1.0 - grid.tint);
    }

    return vec4(radiance, 1.0);
}
