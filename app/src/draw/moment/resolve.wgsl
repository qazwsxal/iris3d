// Resolve pass: absorbance back to transmittance.
//
// At k = 0 the accumulated value is the entire optical depth along the ray, so
// this is exact rather than an approximation — for a single convex volume it is
// the reference image every later moment count has to match.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var moments: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Loaded by integer coordinate, never sampled. Filtering would average the
    // absorbance of neighbouring pixels, whose rays pass through different
    // parts of the volume.
    let accumulated = textureLoad(moments, vec2<i32>(floor(in.position.xy)), 0);

    // A negative total is not physical. It means the front and back faces did
    // not pair up — an open mesh, or geometry clipped by the near plane while
    // the camera sits inside the volume. Clamping keeps the failure to a
    // too-clear volume rather than an `exp` of a large positive number, which
    // would bloom to white.
    let absorbance = max(accumulated.rgb, vec3(0.0));

    // The pipeline blends this multiplicatively: dst = dst * T. Pure absorption
    // dims what is behind and adds nothing of its own.
    return vec4(exp(-absorbance), 1.0);
}
