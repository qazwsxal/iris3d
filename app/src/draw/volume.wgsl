// Ray-marched volume rendering.
//
// Follows what `bevy_pbr`'s volumetric fog does, having got there the hard way.
// Two things matter and both were wrong before:
//
// The transform into the volume comes from the CPU as `uvw_from_world`. An
// earlier version inverted the model matrix here in the fragment shader, which
// meant depending on the mesh instance index surviving into the fragment stage
// to look the matrix up at all. Bevy's own volume shader never does this.
//
// Marching happens in texture space, where the volume is the unit cube. That
// makes the ray-box test a slab test against 0..1 and makes the sample position
// its own texture coordinate, so there is no second mapping to get wrong.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view

struct VolumeUniform {
    // World space into the unit cube the field is stored in.
    uvw_from_world: mat4x4<f32>,
    // x: steps. y: opacity. z: mode, 0 maximum, 1 mean, 2 blend. w: colour map.
    options: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> volume: VolumeUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var field_texture: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var field_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_position = mesh_position_local_to_world(
        get_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * world_position;
    out.world_position = world_position.xyz;
    return out;
}

// Where a ray enters and leaves the unit cube. `near` above `far` means it
// misses.
fn slab(origin: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    // A zero component gives an infinity, which is the answer we want: the ray
    // runs parallel to that pair of planes and never crosses them.
    let inverse_direction = 1.0 / direction;
    let first = -origin * inverse_direction;
    let second = (vec3<f32>(1.0) - origin) * inverse_direction;
    let smaller = min(first, second);
    let larger = max(first, second);
    return vec2<f32>(
        max(max(smaller.x, smaller.y), smaller.z),
        min(min(larger.x, larger.y), larger.z),
    );
}

const VIRIDIS = array<vec3<f32>, 9>(
    vec3<f32>(0.267, 0.005, 0.329),
    vec3<f32>(0.283, 0.141, 0.458),
    vec3<f32>(0.254, 0.265, 0.530),
    vec3<f32>(0.207, 0.372, 0.553),
    vec3<f32>(0.164, 0.471, 0.558),
    vec3<f32>(0.128, 0.567, 0.551),
    vec3<f32>(0.135, 0.659, 0.518),
    vec3<f32>(0.267, 0.749, 0.441),
    vec3<f32>(0.993, 0.906, 0.144),
);

// Matches `draw::sample` on the CPU, including the conversion out of sRGB: the
// stops are quoted in sRGB as they are everywhere else, and the target is a
// linear render.
fn srgb_to_linear(colour: vec3<f32>) -> vec3<f32> {
    let cutoff = step(colour, vec3<f32>(0.04045));
    let low = colour / 12.92;
    let high = pow((colour + 0.055) / 1.055, vec3<f32>(2.4));
    return mix(high, low, cutoff);
}

fn colour_map(map: u32, t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    var rgb: vec3<f32>;
    if map == 0u {
        let scaled = clamped * 8.0;
        let index = min(u32(floor(scaled)), 7u);
        rgb = mix(VIRIDIS[index], VIRIDIS[index + 1u], scaled - f32(index));
    } else if map == 1u {
        let cool = vec3<f32>(0.230, 0.299, 0.754);
        let mid = vec3<f32>(0.865, 0.865, 0.865);
        let warm = vec3<f32>(0.706, 0.016, 0.150);
        if clamped < 0.5 {
            rgb = mix(cool, mid, clamped * 2.0);
        } else {
            rgb = mix(mid, warm, clamped * 2.0 - 1.0);
        }
    } else {
        rgb = vec3<f32>(clamped);
    }
    return srgb_to_linear(rgb);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let uvw_from_world = volume.uvw_from_world;
    let origin = (uvw_from_world * vec4<f32>(view.world_position, 1.0)).xyz;
    // Only the rotation and scale act on a direction, so the translation column
    // is dropped rather than applied.
    let towards = mat3x3<f32>(
        uvw_from_world[0].xyz,
        uvw_from_world[1].xyz,
        uvw_from_world[2].xyz,
    ) * (in.world_position - view.world_position);
    let direction = normalize(towards);

    let span = slab(origin, direction);
    // Start at the camera when it sits inside the cube, not behind it.
    let start = max(span.x, 0.0);
    let stop = span.y;
    if stop <= start {
        discard;
    }

    let steps = max(i32(volume.options.x), 1);
    let step_length = (stop - start) / f32(steps);
    let opacity = volume.options.y;
    let mode = u32(volume.options.z);
    let map = u32(volume.options.w);

    var peak = 0.0;
    var total = 0.0;
    var accumulated = vec3<f32>(0.0);
    var alpha = 0.0;

    // One loop per mode rather than one loop with a branch inside it. The
    // branch is the same for every fragment, so hoisting it costs nothing, and
    // it keeps the blend rule's early exit out of the other two paths.
    if mode == 2u {
        for (var i = 0; i < steps; i = i + 1) {
            let texel = origin + direction * (start + (f32(i) + 0.5) * step_length);
            let value = textureSampleLevel(field_texture, field_sampler, texel, 0.0).r;
            // Front-to-back compositing. Scaling by the step length keeps the
            // result the same when the step count changes, which it must: the
            // step count is a quality control, not a brightness control.
            let sample_alpha = clamp(value * opacity * step_length, 0.0, 1.0);
            accumulated = accumulated + (1.0 - alpha) * colour_map(map, value) * sample_alpha;
            alpha = alpha + (1.0 - alpha) * sample_alpha;
            // Nothing behind an opaque pixel can change it.
            if alpha > 0.995 {
                break;
            }
        }
        if alpha <= 0.0 {
            discard;
        }
        return vec4<f32>(accumulated, alpha);
    }

    if mode == 1u {
        for (var i = 0; i < steps; i = i + 1) {
            let texel = origin + direction * (start + (f32(i) + 0.5) * step_length);
            total = total + textureSampleLevel(field_texture, field_sampler, texel, 0.0).r;
        }
        total = total / f32(steps);
    } else {
        for (var i = 0; i < steps; i = i + 1) {
            // Sample at the middle of each step rather than its edge, so the
            // first and last samples sit inside the volume.
            let texel = origin + direction * (start + (f32(i) + 0.5) * step_length);
            peak = max(peak, textureSampleLevel(field_texture, field_sampler, texel, 0.0).r);
        }
    }

    let value = select(peak, total, mode == 1u);
    // Opacity follows the value, so empty space stays out of the way and what
    // is behind the volume still shows through.
    let shown = clamp(value * opacity, 0.0, 1.0);
    if shown <= 0.0 {
        discard;
    }
    return vec4<f32>(colour_map(map, value), shown);
}
