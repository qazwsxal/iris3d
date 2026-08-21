// Camera-facing quads for point clouds.
//
// Each point becomes four vertices sharing one centre position, with the corner
// offset carried in UV. The offset is applied in *view* space, so the quad is
// always square-on to the camera without the CPU rebuilding anything when the
// view moves.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world}
#import bevy_pbr::mesh_view_bindings::view

// x = quad diameter in world units.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: vec4<f32>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) offset: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);
    let world_position = mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );

    // Offsetting in view space is what makes the quad face the camera: the
    // view-space x/y axes are by definition the screen axes.
    var view_position = (view.view_from_world * world_position).xyz;
    view_position += vec3<f32>(vertex.uv * params.x, 0.0);

    var out: VertexOutput;
    out.clip_position = view.clip_from_view * vec4<f32>(view_position, 1.0);
    out.color = vertex.color;
    out.offset = vertex.uv;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Round the corners off so points read as discs rather than squares.
    // UV runs -0.5..0.5, so the inscribed circle is radius 0.5.
    if dot(in.offset, in.offset) > 0.25 {
        discard;
    }
    return in.color;
}
