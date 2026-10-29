use mikage::ShaderProcessor;

const PHYSICS_TYPES_WGSL: &str = include_str!("../../assets/shaders/physics_types.wgsl");

/// Returns a [`ShaderProcessor`] with `granular_clock::physics_types` pre-registered.
pub fn physics_shader_processor() -> ShaderProcessor {
    let mut sp = ShaderProcessor::new();
    sp.register("granular_clock::physics_types", PHYSICS_TYPES_WGSL);
    sp
}
