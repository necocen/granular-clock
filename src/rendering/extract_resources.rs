//! Shared Main→Render extraction and backend predicates.

use bevy::{
    prelude::*,
    render::extract_resource::{ExtractResource, ExtractResourcePlugin},
};

use crate::simulation::constants::PhysicsBackend;

impl ExtractResource for PhysicsBackend {
    type Source = PhysicsBackend;

    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

/// Backend predicate used by both Main/Render world systems.
pub fn is_cpu_backend(backend: Option<Res<PhysicsBackend>>) -> bool {
    backend
        .as_deref()
        .is_some_and(|backend| *backend == PhysicsBackend::Cpu)
}

/// Backend predicate used by both Main/Render world systems.
pub fn is_gpu_backend(backend: Option<Res<PhysicsBackend>>) -> bool {
    backend
        .as_deref()
        .is_some_and(|backend| *backend == PhysicsBackend::Gpu)
}

/// Main→Render extraction for resources shared across CPU/GPU paths.
pub struct SharedRenderExtractResourcesPlugin;

impl Plugin for SharedRenderExtractResourcesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<PhysicsBackend>::default());
    }
}

/// Registers all Main→Render extraction resources used by rendering plugins.
pub struct RenderExtractResourcesPlugin;

impl Plugin for RenderExtractResourcesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SharedRenderExtractResourcesPlugin);
    }
}
