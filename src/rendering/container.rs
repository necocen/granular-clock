//! Container rendering: floor, divider, and 4 transparent walls.
//!
//! Delegates to [`mikage::SolidRenderer`] for pipeline management and drawing.

use glam::{Mat4, Vec3, Vec4};
use mikage::{SolidObjectId, SolidRenderer};

use crate::simulation::{constants::SimulationConstants, state::SimulationState};

/// Number of container pieces: floor + divider + 4 walls.
const NUM_PIECES: usize = 6;

/// Indices into the pieces array.
const FLOOR: usize = 0;
const DIVIDER: usize = 1;
const WALL_FRONT: usize = 2;
const WALL_BACK: usize = 3;
const WALL_LEFT: usize = 4;
const WALL_RIGHT: usize = 5;

pub struct ContainerRenderer {
    solid: SolidRenderer,
    pieces: [SolidObjectId; NUM_PIECES],
}

impl ContainerRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        view_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mut solid = SolidRenderer::new(device, surface_format, view_bind_group_layout);

        let cube = mikage::CubeMesh::generate();

        let mut pieces = [SolidObjectId::default(); NUM_PIECES];
        for piece in &mut pieces {
            *piece = solid.add_object(device, &cube.positions, &cube.normals, &cube.indices);
        }

        Self { solid, pieces }
    }

    /// Update model matrices each frame based on current simulation state.
    pub fn update_transforms(
        &mut self,
        queue: &wgpu::Queue,
        constants: &SimulationConstants,
        sim_state: &SimulationState,
    ) {
        let container = &constants.container;
        let offset = Vec3::Y * sim_state.container_offset;

        // --- Floor ---
        let floor_translation =
            container.base_position - Vec3::Y * container.half_extents.y + offset;
        let floor_scale = Vec3::new(
            container.half_extents.x * 2.0,
            0.005,
            container.half_extents.z * 2.0,
        );
        let floor_model = Mat4::from_scale_rotation_translation(
            floor_scale,
            glam::Quat::IDENTITY,
            floor_translation,
        );
        self.solid.update_object(
            queue,
            self.pieces[FLOOR],
            floor_model,
            Vec4::new(0.4, 0.35, 0.3, 0.4),
        );

        // --- Divider ---
        let divider_height = container.divider_height;
        let divider_translation = container.base_position - Vec3::Y * container.half_extents.y
            + Vec3::Y * divider_height / 2.0
            + offset;
        let divider_scale = Vec3::new(
            container.divider_thickness,
            divider_height,
            container.half_extents.z * 2.0,
        );
        let divider_model = Mat4::from_scale_rotation_translation(
            divider_scale,
            glam::Quat::IDENTITY,
            divider_translation,
        );
        self.solid.update_object(
            queue,
            self.pieces[DIVIDER],
            divider_model,
            Vec4::new(0.4, 0.4, 0.4, 0.8),
        );

        // --- Walls ---
        let wall_thickness = 0.002;
        let wall_color = Vec4::new(0.5, 0.5, 0.5, 0.05);

        // Front wall (+Z)
        let front_translation = container.base_position + Vec3::Z * container.half_extents.z;
        let front_scale = Vec3::new(
            container.half_extents.x * 2.0,
            container.half_extents.y * 2.0,
            wall_thickness,
        );
        let front_model = Mat4::from_scale_rotation_translation(
            front_scale,
            glam::Quat::IDENTITY,
            front_translation,
        );
        self.solid
            .update_object(queue, self.pieces[WALL_FRONT], front_model, wall_color);

        // Back wall (-Z)
        let back_translation = container.base_position - Vec3::Z * container.half_extents.z;
        let back_model = Mat4::from_scale_rotation_translation(
            front_scale,
            glam::Quat::IDENTITY,
            back_translation,
        );
        self.solid
            .update_object(queue, self.pieces[WALL_BACK], back_model, wall_color);

        // Left wall (-X)
        let left_translation = container.base_position - Vec3::X * container.half_extents.x;
        let lr_scale = Vec3::new(
            wall_thickness,
            container.half_extents.y * 2.0,
            container.half_extents.z * 2.0,
        );
        let left_model =
            Mat4::from_scale_rotation_translation(lr_scale, glam::Quat::IDENTITY, left_translation);
        self.solid
            .update_object(queue, self.pieces[WALL_LEFT], left_model, wall_color);

        // Right wall (+X)
        let right_translation = container.base_position + Vec3::X * container.half_extents.x;
        let right_model = Mat4::from_scale_rotation_translation(
            lr_scale,
            glam::Quat::IDENTITY,
            right_translation,
        );
        self.solid
            .update_object(queue, self.pieces[WALL_RIGHT], right_model, wall_color);
    }

    /// Render all container pieces. Call this within an active render pass.
    ///
    /// The scene bind group must be set at group(0) before calling this.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.solid.render(pass);
    }
}
