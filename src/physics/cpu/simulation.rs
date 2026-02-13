use bevy::prelude::*;
use rayon::prelude::*;

use crate::physics::cpu::contact::{ContactForce, ContactState};
use crate::physics::cpu::{
    clamp_to_container, clamp_velocity, compute_particle_contact_force, compute_wall_contact_force,
    integrate_first_half, integrate_second_half, ContactHistory, SpatialHashGrid,
};
use crate::physics::ParticleStore;
use crate::simulation::{
    constants::{
        advance_oscillation, ContainerParams, MaterialProperties, PhysicsConstants,
        SimulationConstants, WallProperties,
    },
    state::SimulationState,
};

/// 空間ハッシュグリッドを構築
fn build_spatial_grid(grid: &SpatialHashGrid, particles: &ParticleStore) {
    grid.rebuild(
        particles
            .particles
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.position)),
    );
}

/// 全パーティクルの力・トルクをゼロクリア
fn clear_forces(particles: &mut ParticleStore) {
    particles.particles.par_iter_mut().for_each(|p| {
        p.force = Vec3::ZERO;
        p.torque = Vec3::ZERO;
    });
}

/// パーティクル間の衝突を計算
fn compute_particle_collisions(
    particles: &mut ParticleStore,
    grid: &SpatialHashGrid,
    contact_history: &mut ContactHistory,
    material: &MaterialProperties,
    dt: f32,
) {
    #[derive(Clone)]
    struct PairContactResult {
        i: usize,
        j: usize,
        force_i: ContactForce,
        force_j: ContactForce,
        contact_state: ContactState,
    }

    let n = particles.particles.len();
    if n < 2 {
        return;
    }

    // 各粒子について 27 近傍セルのみ探索し、j > i で重複を防ぐ（順序は従来どおり）
    let pairs: Vec<(usize, usize)> = grid.with_cells(|cell_map| {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            let (cx, cy, cz) = grid.cell_index(particles.particles[i].position);
            for &(dx, dy, dz) in SpatialHashGrid::neighbor_offsets() {
                let neighbor_cell = (cx + dx, cy + dy, cz + dz);
                let Some(indices) = cell_map.get(&neighbor_cell) else {
                    continue;
                };

                for &j in indices {
                    if j <= i {
                        continue;
                    }
                    pairs.push((i, j));
                }
            }
        }
        pairs
    });

    // 接触計算本体は並列化。既存接触状態は読み取り専用で参照し、
    // 最終的な加算順序は pairs の順序に揃えて従来の決定性を保つ。
    let pair_results: Vec<PairContactResult> = {
        let previous_contacts = &contact_history.contacts;
        pairs
            .into_par_iter()
            .map(|(i, j)| {
                let p_i = &particles.particles[i];
                let p_j = &particles.particles[j];
                let key = ContactHistory::key(i, j);
                let mut contact_state = previous_contacts.get(&key).cloned().unwrap_or_default();

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
                    &mut contact_state,
                    dt,
                );

                PairContactResult {
                    i,
                    j,
                    force_i,
                    force_j,
                    contact_state,
                }
            })
            .collect()
    };

    let mut force_accum = vec![Vec3::ZERO; n];
    let mut torque_accum = vec![Vec3::ZERO; n];

    for result in pair_results {
        force_accum[result.i] += result.force_i.force;
        torque_accum[result.i] += result.force_i.torque;
        force_accum[result.j] += result.force_j.force;
        torque_accum[result.j] += result.force_j.torque;
        contact_history
            .contacts
            .insert((result.i, result.j), result.contact_state);
    }

    for i in 0..n {
        particles.particles[i].force += force_accum[i];
        particles.particles[i].torque += torque_accum[i];
    }
}

/// 壁との衝突を計算
fn compute_wall_collisions(
    particles: &mut ParticleStore,
    container_params: &ContainerParams,
    container_offset: f32,
    wall_props: &WallProperties,
) {
    particles.particles.par_iter_mut().for_each(|p| {
        let wall_force = compute_wall_contact_force(
            p.position,
            p.velocity,
            p.angular_velocity,
            p.radius,
            p.mass,
            container_params,
            container_offset,
            wall_props,
        );
        p.force += wall_force.force;
        p.torque += wall_force.torque;
    });
}

/// 位置の積分（Velocity Verlet 前半ステップ）
fn integrate_positions(particles: &mut ParticleStore, physics: &PhysicsConstants, dt: f32) {
    let gravity = physics.gravity;

    particles.particles.par_iter_mut().for_each(|p| {
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
    });
}

/// 速度の積分（Velocity Verlet 後半ステップ）
fn integrate_velocities(particles: &mut ParticleStore, physics: &PhysicsConstants, dt: f32) {
    let gravity = physics.gravity;

    particles.particles.par_iter_mut().for_each(|p| {
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
    });
}

/// パーティクルをコンテナ内にクランプ
fn clamp_particles(
    particles: &mut ParticleStore,
    container_params: &ContainerParams,
    container_offset: f32,
) {
    let box_offset = Vec3::Y * container_offset;
    let box_min = container_params.base_position - container_params.half_extents + box_offset;
    let box_max = container_params.base_position + container_params.half_extents + box_offset;

    const MAX_LINEAR_VEL: f32 = 10.0;
    const MAX_ANGULAR_VEL: f32 = 100.0;

    particles.particles.par_iter_mut().for_each(|p| {
        clamp_to_container(&mut p.position, &mut p.velocity, p.radius, box_min, box_max);
        clamp_velocity(
            &mut p.velocity,
            &mut p.angular_velocity,
            MAX_LINEAR_VEL,
            MAX_ANGULAR_VEL,
        );
    });
}

/// 物理サブステップを実行するシステム（exclusive system）
pub fn run_physics_substeps(world: &mut World) {
    // 一時停止中なら何もしない
    let paused = world.resource::<SimulationState>().paused;
    if paused {
        return;
    }

    let constants = world.resource::<SimulationConstants>().clone();
    let substeps = constants.settings.substeps_per_frame;

    // リソースを一括取り出し（exclusive system なので安全）
    let mut particles = world.remove_resource::<ParticleStore>().unwrap();
    let mut contact_history = world.remove_resource::<ContactHistory>().unwrap();
    let mut sim_state = world.remove_resource::<SimulationState>().unwrap();

    // 不変リソースはコピー（Copy 型なので安全かつ参照の衝突を回避）
    let grid = world.resource::<SpatialHashGrid>();
    let material = constants.material;
    let physics = constants.physics;
    let wall_props = constants.wall;
    let container_params = &constants.container;
    let osc_params = &constants.oscillation;
    let dt = constants.time.dt;

    for _ in 0..substeps {
        advance_oscillation(&mut sim_state, osc_params, dt);
        build_spatial_grid(grid, &particles);
        clear_forces(&mut particles);
        compute_particle_collisions(&mut particles, grid, &mut contact_history, &material, dt);
        compute_wall_collisions(
            &mut particles,
            container_params,
            sim_state.container_offset,
            &wall_props,
        );
        integrate_positions(&mut particles, &physics, dt);
        clamp_particles(&mut particles, container_params, sim_state.container_offset);
        // 後半ステップの力計算は更新後の位置に対して行う
        build_spatial_grid(grid, &particles);
        clear_forces(&mut particles);
        compute_particle_collisions(&mut particles, grid, &mut contact_history, &material, dt);
        compute_wall_collisions(
            &mut particles,
            container_params,
            sim_state.container_offset,
            &wall_props,
        );
        integrate_velocities(&mut particles, &physics, dt);
        clamp_particles(&mut particles, container_params, sim_state.container_offset);
        contact_history.cleanup();
        sim_state.step_time(dt);
    }

    // リソースを戻す
    world.insert_resource(particles);
    world.insert_resource(contact_history);
    world.insert_resource(sim_state);
}
