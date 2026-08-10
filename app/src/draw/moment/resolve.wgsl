// Resolve pass: absorbance back to transmittance.
//
// This applies the *total* absorbance rather than reconstructing from the
// moments, and that is the accurate choice rather than a shortcut.
//
// The accumulation clamps both endpoints of every interval to the opaque depth
// (see moment.wgsl), so nothing is deposited behind the surface being lit. The
// fraction of the absorbance lying in front of that surface is therefore
// exactly one, and the total is already the answer. Asking the moments the same
// question can only lose: the reconstruction returns a lower bound, and at the
// far end of the domain that bound falls short of one — by around 7% for a
// volume spanning most of the depth range, which showed up as volumes
// rendering visibly too bright.
//
// The moments are not wasted. They answer the question this pass cannot ask:
// how much absorbance lies in front of some depth *other* than the opaque
// surface. That is what a transparent actor needs in order to attenuate itself,
// and it is the second geometry pass of §2. See reconstruct.wgsl.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Only the totals are read here. The layout still carries the view, the
// moments, the depth buffer and the depth bound, because it is shared with the
// pass that will need them; a bind group entry the shader does not declare is
// legal and costs nothing.
@group(0) @binding(2) var totals_texture: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Loaded by integer coordinate, never sampled. Filtering would average the
    // absorbance of neighbouring pixels, whose rays pass through different
    // parts of the volume.
    let coord = vec2<i32>(floor(in.position.xy));
    let totals = textureLoad(totals_texture, coord, 0);

    // A negative total is not physical. It means the front and back faces did
    // not pair up — an open mesh, or geometry clipped by the near plane while
    // the camera sits inside the volume. Clamping keeps the failure to a
    // too-clear volume rather than an `exp` of a large positive number, which
    // would bloom to white.
    let absorbance = max(totals.rgb, vec3(0.0));

    // The pipeline blends this multiplicatively: dst = dst * T. Pure absorption
    // dims what is behind and adds nothing of its own.
    return vec4(exp(-absorbance), 1.0);
}
