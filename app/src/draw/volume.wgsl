// Ray-marched volume rendering.
//
// The mesh is the grid's bounding box, wound *inside out*. With ordinary back
// face culling that means the fragment we get is the point where the ray
// leaves the box, and it works unchanged when the camera is inside the box —
// which is the case an outward-wound box gets wrong. It also saves specialising
// the pipeline to flip the cull mode.
//
// Everything below happens in the object's local space. The camera arrives in
// world space, so it is brought back through the inverse of the model matrix;
// that keeps the volume correct under rotation and scale without the CPU
// resending anything when the object moves.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view

struct VolumeUniform {
    // xyz: low corner of the grid in local space. w: number of steps.
    bounds_min: vec4<f32>,
    // xyz: size of the grid in local space. w: opacity scale.
    bounds_size: vec4<f32>,
    // x: mode, 0 maximum, 1 mean, 2 blend. y: colour map.
    options: vec4<f32>,
    // xyz: sample counts along each axis.
    dims: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> volume: VolumeUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var field_texture: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var field_sampler: sampler;

// Fetched by integer coordinate rather than sampled.
//
// `textureSampleLevel` is legal in non-uniform control flow, but the marching
// loop reaches it after a conditional break and the result was undefined in
// practice — whole fragments came back as noise. `textureLoad` has no such
// rule, and the cost is nearest-neighbour sampling instead of linear, which a
// first pass can live with.
fn field_at(texel: vec3<f32>) -> f32 {
    let dims = volume.dims.xyz;
    let clamped = clamp(texel, vec3<f32>(0.0), vec3<f32>(1.0));
    let coordinate = min(vec3<i32>(clamped * dims), vec3<i32>(dims - 1.0));
    return textureLoad(field_texture, coordinate, 0).r;
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_position: vec3<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);
    let world_position = mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * world_position;
    out.local_position = vertex.position;
    out.instance_index = vertex.instance_index;
    return out;
}

// Inverse of a 3x3, by the adjugate over the determinant. The model matrix is
// affine, so inverting its rotation-and-scale part and the translation
// separately is enough — and avoids a full 4x4 inverse.
fn inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let det = dot(a, r0);
    return transpose(mat3x3<f32>(r0 / det, r1 / det, r2 / det));
}

// Where a ray enters and leaves an axis-aligned box. Returns (near, far), and
// near is above far when the ray misses.
fn slab(origin: vec3<f32>, direction: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> vec2<f32> {
    // A zero component gives an infinity here, which is the answer we want: the
    // ray is parallel to that pair of planes and never crosses them.
    let inverse_direction = 1.0 / direction;
    let first = (low - origin) * inverse_direction;
    let second = (high - origin) * inverse_direction;
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
    let world_from_local = get_world_from_local(in.instance_index);
    let rotation = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    );
    let local_from_world = inverse3(rotation);
    let camera = local_from_world * (view.world_position - world_from_local[3].xyz);

    let low = volume.bounds_min.xyz;
    let size = volume.bounds_size.xyz;
    let high = low + size;

    let exit = in.local_position;
    let direction = normalize(exit - camera);
    let span = slab(camera, direction, low, high);

    // Start at the camera when it sits inside the box, not behind it.
    let start = max(span.x, 0.0);
    let stop = min(span.y, length(exit - camera));
    if stop <= start {
        discard;
    }

    let steps = max(i32(volume.bounds_min.w), 1);
    let step_length = (stop - start) / f32(steps);
    let mode = u32(volume.options.x);
    let map = u32(volume.options.y);
    let opacity = volume.bounds_size.w;

    var peak = 0.0;
    var total = 0.0;
    var accumulated = vec3<f32>(0.0);
    var alpha = 0.0;

    // One loop per mode rather than one loop with a branch inside it. The
    // branch is the same for every fragment, so hoisting it costs nothing, and
    // it keeps the blend rule's early exit out of the other two paths.
    if mode == 2u {
        for (var i = 0; i < steps; i = i + 1) {
            let distance = start + (f32(i) + 0.5) * step_length;
            let value = field_at((camera + direction * distance - low) / size);
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
    } else if mode == 1u {
        for (var i = 0; i < steps; i = i + 1) {
            let distance = start + (f32(i) + 0.5) * step_length;
            total = total + field_at((camera + direction * distance - low) / size);
        }
    } else {
        for (var i = 0; i < steps; i = i + 1) {
            // Sample at the middle of each step rather than its edge, so the
            // first and last samples sit inside the volume.
            let distance = start + (f32(i) + 0.5) * step_length;
            peak = max(peak, field_at((camera + direction * distance - low) / size));
        }
    }

    if mode == 2u {
        if alpha <= 0.0 {
            discard;
        }
        return vec4<f32>(accumulated, alpha);
    }

    var value = peak;
    if mode == 1u {
        value = total / f32(steps);
    }
    // Opacity follows the value, so empty space stays out of the way and what
    // is behind the volume still shows through.
    let shown = clamp(value * opacity, 0.0, 1.0);
    if shown <= 0.0 {
        discard;
    }
    return vec4<f32>(colour_map(map, value), shown);
}
