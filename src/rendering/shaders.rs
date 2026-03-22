/// Preprocess a compute shader by prepending physics_types.wgsl content.
/// Removes #import and #define_import_path lines.
pub fn preprocess_compute_shader(main_source: &str) -> String {
    let types = include_str!("../../assets/shaders/physics_types.wgsl");
    let types_clean: String = types
        .lines()
        .filter(|l| !l.starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n");
    let main_clean: String = main_source
        .lines()
        .filter(|l| !l.starts_with("#import"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{}\n{}", types_clean, main_clean)
}
