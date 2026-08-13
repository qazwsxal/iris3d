// The dielectric shell: the surface the absorbing interior does not have.
//
// Specular only. Nothing here absorbs, nothing here blocks, and the whole pass
// is additive — so it stays order-independent like the accumulation, and both
// faces of a closed mesh contribute in whatever order they arrive.
//
// Both faces contributing is deliberate rather than an oversight. A thin glass
// shell really does reflect at the far interface as well as the near one, and
// at grazing angles that second rim is a large part of what makes a shape read
// as hollow rather than solid. Culling would remove it and leave the silhouette
// looking painted on.

#import bevy_render::view::View
#import iris3d::moment_reconstruct::absorbed_fraction

// Must match `prepare::MomentInstance` and the copy in moment.wgsl field for
// field. One buffer serves both passes, and a mismatch does not fail to
// compile — it silently reads a field out of another field's bytes.
struct Instance {
    world_from_local: mat4x4<f32>,
    world_from_local_normal: mat3x3<f32>,
    tint: vec3<f32>,
    strength: f32,
    dirac: u32,
    f0: f32,
    roughness: f32,
}

struct Light {
    towards: vec3<f32>,
    intensity: f32,
    colour: vec3<f32>,
    _pad: f32,
}

struct Lighting {
    lights: array<Light, 4>,
    background: vec3<f32>,
    count: u32,
}

struct Bounds {
    near: f32,
    far: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<uniform> lighting: Lighting;
#ifdef MULTISAMPLED
@group(0) @binding(3) var moments_texture: texture_multisampled_2d<f32>;
@group(0) @binding(4) var totals_texture: texture_multisampled_2d<f32>;
#else
@group(0) @binding(3) var moments_texture: texture_2d<f32>;
@group(0) @binding(4) var totals_texture: texture_2d<f32>;
#endif
@group(0) @binding(5) var<uniform> bounds: Bounds;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) @interpolate(flat) instance: u32,
    // View-space distance in front of the camera, as in moment.wgsl: the moment
    // domain is built from a linear depth, not from reverse-Z clip space.
    @location(3) view_z: f32,
}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) local_position: vec3<f32>,
    @location(1) local_normal: vec3<f32>,
) -> VertexOutput {
    let instance = instances[instance_index];
    let world = instance.world_from_local * vec4(local_position, 1.0);

    var out: VertexOutput;
    out.position = view.clip_from_world * world;
    out.world_position = world.xyz;
    // The inverse transpose, so a non-uniformly scaled object still has normals
    // perpendicular to its surface rather than merely to its unscaled one.
    out.world_normal = instance.world_from_local_normal * local_normal;
    out.instance = instance_index;
    out.view_z = -(view.view_from_world * world).z;
    return out;
}

// How much of what this surface reflects survives the medium in front of it.
//
// This is the question the moments were accumulated to answer, and the reason
// four of them are stored rather than the total alone: the total says how much
// absorbance lies along the whole ray, and a back face needs to know how much
// lies in front of *it*. `absorbed_fraction` reconstructs that from the depth
// distribution.
//
// Applied to the per-channel totals rather than reconstructed per channel,
// which is the assumption the two-attachment split rests on: a volume's depth
// structure is the same in every channel while its strength differs.
//
// The bound is loose for a continuous measure — see reconstruct.wgsl, where a
// uniform slab queried at its own midpoint comes back 0.26 against a true 0.50
// — and it is a *lower* bound, so the error is always towards too little
// attenuation. A far highlight therefore still reads brighter than it should,
// just no longer as bright as a near one.
fn transmittance_in_front(coord: vec2<i32>, w: f32) -> vec3<f32> {
#ifdef MULTISAMPLED
    let totals = textureLoad(totals_texture, coord, 0);
    let moments = textureLoad(moments_texture, coord, 0);
#else
    let totals = textureLoad(totals_texture, coord, 0);
    let moments = textureLoad(moments_texture, coord, 0);
#endif

    // b0 of the scalar measure. At zero nothing was accumulated on this pixel,
    // so nothing is in front of anything and the surface is seen directly.
    let total = totals.a;
    if total <= 1e-5 {
        return vec3(1.0);
    }

    let fraction = absorbed_fraction(moments / total, w);
    return exp(-max(totals.rgb, vec3(0.0)) * fraction);
}

// How reflective the surface is allowed to become at grazing incidence.
//
// A real dielectric reaches a full mirror at ninety degrees, and Schlick's
// approximation says so by driving the term to 1. That is right for a surface
// standing in a real environment, where the grazing reflection is *of*
// something. Here it is of a synthetic gradient, with no horizon occlusion and
// no contact shadowing, so a ceiling of 1 reads as a glowing outline traced
// around the silhouette rather than as a reflection.
//
// This is a look control, not physics, and it is the knob to reach for if the
// rim is too hot. Acrylic and glass differ far less than you would guess —
// their reflectance straight on is 0.039 against 0.043 — so the rim, not the
// index, is what separates the two by eye.
const GRAZING_CEILING: f32 = 0.55;

// Schlick's approximation, with that ceiling in place of the usual 1.
fn fresnel(f0: f32, cos_theta: f32) -> f32 {
    return f0 + (GRAZING_CEILING - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

// GGX / Trowbridge-Reitz normal distribution.
fn distribution(alpha: f32, n_dot_h: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-6);
}

// Smith's height-correlated visibility term, already divided by the
// `4 * n_dot_l * n_dot_v` a specular BRDF would otherwise carry.
fn visibility(alpha: f32, n_dot_v: f32, n_dot_l: f32) -> f32 {
    let a2 = alpha * alpha;
    let v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(v + l, 1e-6);
}

// What the object mirrors, taken from the viewport's own background.
//
// The scene really is a uniform void of one colour, so that colour is what a
// dielectric in it reflects, and changing the viewport's background now changes
// the glass with it. It replaces an invented sky-to-ground gradient that looked
// convincing and was answering for an environment that is not there — worse,
// its ground was eight times darker than its sky, so the shell's brightness
// swung with the camera's elevation for no reason in the scene.
//
// The remaining gradient is a deliberate legibility cue rather than physics: a
// perfectly uniform environment reflects identically everywhere, which leaves a
// curved surface with no shading to read its shape from. Kept mild, and
// centred on the background so the mean is still what is actually behind the
// object.
const ENVIRONMENT_FLOOR: f32 = 0.55;
const ENVIRONMENT_CEILING: f32 = 1.85;

fn environment(direction: vec3<f32>) -> vec3<f32> {
    let up = saturate(direction.y * 0.5 + 0.5);
    return lighting.background * mix(ENVIRONMENT_FLOOR, ENVIRONMENT_CEILING, up);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let instance = instances[in.instance];

    let v = normalize(view.world_position - in.world_position);
    // Both faces are drawn, and a reflection is about the angle to the surface
    // rather than which side of it we are on, so a normal pointing away from
    // the viewer is flipped rather than the fragment discarded.
    //
    // Keyed on the geometry, **not** on `@builtin(front_facing)`. Winding and
    // vertex normals are two independent conventions, and a client's mesh is
    // free to disagree about them — a swept tube built on a transported frame
    // routinely does. Flipping on winding then turns the normal away from the
    // viewer on every visible face, which mirrors `reflected` and samples the
    // environment gradient upside down: the shell goes dark seen from above and
    // bright seen from below, which is exactly backwards.
    var n = normalize(in.world_normal);
    if dot(n, v) < 0.0 {
        n = -n;
    }

    let n_dot_v = saturate(dot(n, v));
    let reflected = reflect(-v, n);
    // Squared, which is the usual mapping from a roughness a person sets to the
    // alpha a GGX lobe wants. Floored so a mirror-smooth shell still has a
    // highlight of non-zero width rather than one that falls between pixels.
    let alpha = max(instance.roughness * instance.roughness, 1e-3);
    let f = fresnel(instance.f0, n_dot_v);

    var specular = vec3(0.0);
    for (var i = 0u; i < min(lighting.count, 4u); i = i + 1u) {
        let light = lighting.lights[i];
        let l = normalize(light.towards);
        let n_dot_l = saturate(dot(n, l));
        if n_dot_l <= 0.0 {
            continue;
        }
        let h = normalize(l + v);
        let n_dot_h = saturate(dot(n, h));
        let d = distribution(alpha, n_dot_h);
        let vis = visibility(alpha, n_dot_v, n_dot_l);
        specular += light.colour * light.intensity * d * vis * n_dot_l;
    }

    // The environment answers for everything the handful of lights does not,
    // and is what carries the grazing-angle rim.
    specular += environment(reflected);

    // Everything above is what the surface reflects. What reaches the eye is
    // what survives the medium between here and the camera — which for the far
    // wall of a hollow shape is the whole thickness of the near wall, and for
    // the near wall is nothing at all. Without this every back face reads as
    // bright as a front one and the shape looks like frosted glass rather than
    // something with an inside.
    let span = max(bounds.far - bounds.near, 1e-6);
    let w = clamp((in.view_z - bounds.near) / span, 0.0, 1.0);
    let survives = transmittance_in_front(vec2<i32>(floor(in.position.xy)), w);

    // Additive: this is light the surface sends towards the eye, on top of
    // whatever the interior let through.
    return vec4(specular * f * survives, 1.0);
}
