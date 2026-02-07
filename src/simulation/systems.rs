use bevy::prelude::*;

use crate::physics::{
    clamp_to_container, clamp_velocity, compute_particle_contact_force, compute_wall_contact_force,
    integrate_first_half, integrate_second_half, ContactHistory, MaterialProperties, ParticleStore,
    PhysicsConstants, SpatialHashGrid, WallProperties,
};

use super::{Container, OscillationParams, SimulationSettings, SimulationState, SimulationTime};

/// 振動を更新
fn update_oscillation(
    container: &mut Container,
    params: &mut OscillationParams,
    sim_time: &SimulationTime,
) {
    use std::f32::consts::PI;
    if !params.enabled {
        container.current_offset = 0.0;
        return;
    }
    params.phase += params.frequency * 2.0 * PI * sim_time.dt;
    if params.phase > 2.0 * PI {
        params.phase -= 2.0 * PI;
    }
    container.current_offset = params.amplitude * params.phase.sin();
}

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

    // 衝突ペアを収集
    let mut collision_pairs: Vec<(usize, usize)> = Vec::new();
    for bucket_idx in 0..grid.table_size {
        let bucket = grid.buckets[bucket_idx].lock();
        let indices: Vec<usize> = bucket.clone();
        drop(bucket);

        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                collision_pairs.push((indices[i], indices[j]));
            }
        }
    }

    // 衝突力を計算して蓄積
    for (i1, i2) in collision_pairs {
        // split_at_mut で安全に 2 つの要素への可変参照を取得
        let (lo, hi) = if i1 < i2 { (i1, i2) } else { (i2, i1) };
        let (left, right) = particles.particles.split_at_mut(hi);
        let p_lo = &left[lo];
        let p_hi = &right[0];

        let contact_state = contact_history.get_or_create(i1, i2);

        let (force_lo, force_hi) = compute_particle_contact_force(
            p_lo.position,
            p_lo.velocity,
            p_lo.angular_velocity,
            p_lo.radius,
            p_lo.mass,
            p_hi.position,
            p_hi.velocity,
            p_hi.angular_velocity,
            p_hi.radius,
            p_hi.mass,
            material,
            contact_state,
            dt,
        );

        // i1 < i2 のとき lo=i1, hi=i2 なので force_lo→i1, force_hi→i2
        // i1 > i2 のとき lo=i2, hi=i1 なので force_lo→i2, force_hi→i1
        if i1 < i2 {
            particles.particles[i1].force += force_lo.force;
            particles.particles[i1].torque += force_lo.torque;
            particles.particles[i2].force += force_hi.force;
            particles.particles[i2].torque += force_hi.torque;
        } else {
            particles.particles[i2].force += force_lo.force;
            particles.particles[i2].torque += force_lo.torque;
            particles.particles[i1].force += force_hi.force;
            particles.particles[i1].torque += force_hi.torque;
        }
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
        update_oscillation(&mut container, &mut osc_params, &sim_time);
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
