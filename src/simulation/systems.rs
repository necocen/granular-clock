use bevy::prelude::*;
use std::collections::HashMap;

use crate::physics::{
    clamp_to_container, clamp_velocity, compute_particle_contact_force, compute_wall_contact_force,
    integrate_first_half, integrate_second_half, ContactHistory, MaterialProperties, ParticleStore,
    PhysicsConstants, SpatialHashGrid, WallProperties,
};

use super::{
    advance_oscillation, Container, OscillationParams, SimulationSettings, SimulationState,
    SimulationTime,
};

/// 空間ハッシュグリッドを構築
fn build_spatial_grid(grid: &SpatialHashGrid, particles: &ParticleStore) {
    grid.clear();
    for (i, p) in particles.particles.iter().enumerate() {
        grid.insert(i, p.position);
    }
}

/// 全パーティクルの力・トルクをゼロクリア
fn clear_forces(particles: &mut ParticleStore) {
    for p in particles.particles.iter_mut() {
        p.force = Vec3::ZERO;
        p.torque = Vec3::ZERO;
    }
}

/// パーティクル間の衝突を計算
fn compute_particle_collisions(
    particles: &mut ParticleStore,
    grid: &SpatialHashGrid,
    contact_history: &mut ContactHistory,
    material: &MaterialProperties,
    sim_time: &SimulationTime,
) {
    let dt = sim_time.dt;
    let n = particles.particles.len();
    if n < 2 {
        return;
    }

    // セル -> 粒子インデックス群を構築（正確な近傍探索用）
    let mut cells = Vec::with_capacity(n);
    let mut cell_map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(n);
    for (i, p) in particles.particles.iter().enumerate() {
        let cell = (
            (p.position.x / grid.cell_size).floor() as i32,
            (p.position.y / grid.cell_size).floor() as i32,
            (p.position.z / grid.cell_size).floor() as i32,
        );
        cells.push(cell);
        cell_map.entry(cell).or_default().push(i);
    }

    let mut force_accum = vec![Vec3::ZERO; n];
    let mut torque_accum = vec![Vec3::ZERO; n];

    // 各粒子について 27 近傍セルのみ探索し、j > i で重複を防ぐ
    for i in 0..n {
        let (cx, cy, cz) = cells[i];
        let p_i = &particles.particles[i];

        for &(dx, dy, dz) in SpatialHashGrid::neighbor_offsets() {
            let neighbor_cell = (cx + dx, cy + dy, cz + dz);
            let Some(indices) = cell_map.get(&neighbor_cell) else {
                continue;
            };

            for &j in indices {
                if j <= i {
                    continue;
                }

                let p_j = &particles.particles[j];
                let contact_state = contact_history.get_or_create(i, j);

                let (force_i, force_j) = compute_particle_contact_force(
                    p_i.position,
                    p_i.velocity,
                    p_i.angular_velocity,
                    p_i.radius,
                    p_i.mass,
                    p_j.position,
                    p_j.velocity,
                    p_j.angular_velocity,
                    p_j.radius,
                    p_j.mass,
                    material,
                    contact_state,
                    dt,
                );

                force_accum[i] += force_i.force;
                torque_accum[i] += force_i.torque;
                force_accum[j] += force_j.force;
                torque_accum[j] += force_j.torque;
            }
        }
    }

    for i in 0..n {
        particles.particles[i].force += force_accum[i];
        particles.particles[i].torque += torque_accum[i];
    }
}

/// 壁との衝突を計算
fn compute_wall_collisions(
    particles: &mut ParticleStore,
    container: &Container,
    wall_props: &WallProperties,
) {
    for p in particles.particles.iter_mut() {
        let wall_force = compute_wall_contact_force(
            p.position,
            p.velocity,
            p.angular_velocity,
            p.radius,
            p.mass,
            container,
            wall_props,
        );
        p.force += wall_force.force;
        p.torque += wall_force.torque;
    }
}

/// 位置の積分（Velocity Verlet 前半ステップ）
fn integrate_positions(
    particles: &mut ParticleStore,
    physics: &PhysicsConstants,
    sim_time: &SimulationTime,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    for p in particles.particles.iter_mut() {
        integrate_first_half(
            &mut p.position,
            &mut p.velocity,
            &mut p.angular_velocity,
            p.force,
            p.torque,
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
    }
}

/// 速度の積分（Velocity Verlet 後半ステップ）
fn integrate_velocities(
    particles: &mut ParticleStore,
    physics: &PhysicsConstants,
    sim_time: &SimulationTime,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    for p in particles.particles.iter_mut() {
        integrate_second_half(
            &mut p.velocity,
            &mut p.angular_velocity,
            p.force,
            p.torque,
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
    }
}

/// パーティクルをコンテナ内にクランプ
fn clamp_particles(particles: &mut ParticleStore, container: &Container) {
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;

    const MAX_LINEAR_VEL: f32 = 10.0;
    const MAX_ANGULAR_VEL: f32 = 100.0;

    for p in particles.particles.iter_mut() {
        clamp_to_container(&mut p.position, &mut p.velocity, p.radius, box_min, box_max);
        clamp_velocity(
            &mut p.velocity,
            &mut p.angular_velocity,
            MAX_LINEAR_VEL,
            MAX_ANGULAR_VEL,
        );
    }
}

/// 物理サブステップを実行するシステム（exclusive system）
pub fn run_physics_substeps(world: &mut World) {
    // 一時停止中なら何もしない
    let paused = world.resource::<SimulationState>().paused;
    if paused {
        return;
    }

    let substeps = world.resource::<SimulationSettings>().substeps_per_frame;

    // リソースを一括取り出し（exclusive system なので安全）
    let mut particles = world.remove_resource::<ParticleStore>().unwrap();
    let mut contact_history = world.remove_resource::<ContactHistory>().unwrap();
    let mut container = world.remove_resource::<Container>().unwrap();
    let mut osc_params = world.remove_resource::<OscillationParams>().unwrap();
    let mut sim_time = world.remove_resource::<SimulationTime>().unwrap();

    // 不変リソースはコピー（Copy 型なので安全かつ参照の衝突を回避）
    let grid = world.resource::<SpatialHashGrid>();
    let material = *world.resource::<MaterialProperties>();
    let physics = *world.resource::<PhysicsConstants>();
    let wall_props = *world.resource::<WallProperties>();

    for _ in 0..substeps {
        advance_oscillation(&mut container, &mut osc_params, sim_time.dt);
        build_spatial_grid(grid, &particles);
        clear_forces(&mut particles);
        compute_particle_collisions(
            &mut particles,
            grid,
            &mut contact_history,
            &material,
            &sim_time,
        );
        compute_wall_collisions(&mut particles, &container, &wall_props);
        integrate_positions(&mut particles, &physics, &sim_time);
        clamp_particles(&mut particles, &container);
        // 後半ステップの力計算は更新後の位置に対して行う
        build_spatial_grid(grid, &particles);
        clear_forces(&mut particles);
        compute_particle_collisions(
            &mut particles,
            grid,
            &mut contact_history,
            &material,
            &sim_time,
        );
        compute_wall_collisions(&mut particles, &container, &wall_props);
        integrate_velocities(&mut particles, &physics, &sim_time);
        clamp_particles(&mut particles, &container);
        contact_history.cleanup();
        sim_time.step();
    }

    // リソースを戻す
    world.insert_resource(particles);
    world.insert_resource(contact_history);
    world.insert_resource(container);
    world.insert_resource(osc_params);
    world.insert_resource(sim_time);
}
