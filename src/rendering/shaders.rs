use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    shader::Shader,
};

#[derive(Resource, Default)]
struct RenderingEmbeddedShadersLoaded;

pub const PARTICLE_INSTANCING_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("6dfc52b8-8c29-4f40-a45e-5468686e6ff7");

pub fn load_rendering_internal_shaders(app: &mut App) {
    if app
        .world()
        .contains_resource::<RenderingEmbeddedShadersLoaded>()
    {
        return;
    }

    load_internal_asset!(
        app,
        PARTICLE_INSTANCING_SHADER_HANDLE,
        "../../assets/shaders/particle_instancing.wgsl",
        Shader::from_wgsl
    );

    app.insert_resource(RenderingEmbeddedShadersLoaded);
}
