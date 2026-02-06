// Instanced particle rendering shader
// GPU インスタンスバッファから直接粒子を描画する
//
// Vertex buffer 0: sphere mesh (position, normal, uv)
// Vertex buffer 1: instance data (pos_scale, color) - per instance

#import bevy_pbr::mesh_view_bindings::view

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,

    // Per-instance attributes (from instance buffer)
    @location(3) i_pos_scale: vec4<f32>,  // xyz = world position, w = radius
    @location(4) i_color: vec4<f32>,      // RGBA
};

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex(v: Vertex) -> VertexOut {
    // Scale sphere vertex by radius and translate to particle world position
    let world_pos = v.position * v.i_pos_scale.w + v.i_pos_scale.xyz;

    var out: VertexOut;
    // Transform directly from world space to clip space
    out.clip_position = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    out.color = v.i_color;
    // Normal is uniform-scaled so no correction needed
    out.world_normal = v.normal;
    return out;
}

@fragment
fn fragment(in: VertexOut) -> @location(0) vec4<f32> {
    // Simple Lambert diffuse + ambient lighting
    let light_dir = normalize(vec3<f32>(1.0, 2.0, 1.0));
    let n = normalize(in.world_normal);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.3;
    let lit = ambient + diffuse * 0.7;

    return vec4<f32>(in.color.rgb * lit, in.color.a);
}
