use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    shader::Shader,
};

#[derive(Resource, Default)]
struct GpuEmbeddedShadersLoaded;

pub const PHYSICS_TYPES_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("17e4a084-5b16-4f7b-b1b0-f43ee0f46ea7");
pub const HASH_KEYS_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("81320624-cb10-4d38-8da8-67d8cc995e58");
pub const BITONIC_SORT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("77fce7ef-d344-4209-b3f4-a65dad25e133");
pub const CELL_RANGES_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f85fe6bf-5166-4ea8-914b-12ac4eb743e8");
pub const COLLISION_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("f47a79a9-fa8b-4531-a595-e8ac7f09459e");
pub const INTEGRATE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("0b75dbbe-e58b-4f73-b80f-0d5a72a926fe");
pub const PARTICLE_TO_INSTANCE_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("454eaac8-bf62-42cf-a2e3-a630addd5cc2");

pub fn load_gpu_internal_shaders(app: &mut App) {
    if app.world().contains_resource::<GpuEmbeddedShadersLoaded>() {
        return;
    }

    load_internal_asset!(
        app,
        PHYSICS_TYPES_SHADER_HANDLE,
        "../../../assets/shaders/physics_types.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        HASH_KEYS_SHADER_HANDLE,
        "../../../assets/shaders/hash_keys.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        BITONIC_SORT_SHADER_HANDLE,
        "../../../assets/shaders/bitonic_sort.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        CELL_RANGES_SHADER_HANDLE,
        "../../../assets/shaders/cell_ranges.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        COLLISION_SHADER_HANDLE,
        "../../../assets/shaders/collision.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        INTEGRATE_SHADER_HANDLE,
        "../../../assets/shaders/integrate.wgsl",
        Shader::from_wgsl
    );
    load_internal_asset!(
        app,
        PARTICLE_TO_INSTANCE_SHADER_HANDLE,
        "../../../assets/shaders/particle_to_instance.wgsl",
        Shader::from_wgsl
    );

    app.insert_resource(GpuEmbeddedShadersLoaded);
}
