use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;

use crate::physics::{
    clamp_to_container, clamp_velocity, compute_particle_contact_force, compute_wall_contact_force,
    integrate_first_half, integrate_second_half, AngularVelocity, ContactHistory, Force,
    MaterialProperties, ParticleProperties, PhysicsConstants, Position, SpatialHashGrid, Torque,
    Velocity, WallProperties,
};

use super::{Container, SimulationSettings, SimulationTime};

/// 空間ハッシュグリッドを構築
pub fn build_spatial_grid(
    grid: Res<SpatialHashGrid>,
    particles: Query<(Entity, &Position)>,
) {
    // グリッドをクリア
    grid.clear();

    // 粒子を挿入
    for (entity, pos) in particles.iter() {
        grid.insert(entity, pos.0);
    }
}

/// 力とトルクをリセット
pub fn clear_forces(mut particles: Query<(&mut Force, &mut Torque)>) {
    particles.par_iter_mut().for_each(|(mut force, mut torque)| {
        force.0 = Vec3::ZERO;
        torque.0 = Vec3::ZERO;
    });
}

/// 粒子間の衝突を計算
pub fn compute_particle_collisions(
    grid: Res<SpatialHashGrid>,
    mut particles: Query<(
        Entity,
        &Position,
        &Velocity,
        &AngularVelocity,
        &mut Force,
        &mut Torque,
        &ParticleProperties,
    )>,
    mut contact_history: ResMut<ContactHistory>,
    material: Res<MaterialProperties>,
    sim_time: Res<SimulationTime>,
) {
    let dt = sim_time.dt;

    // 衝突ペアを収集
    let mut collision_pairs: Vec<(Entity, Entity)> = Vec::new();

    // 全バケットを走査して衝突ペアを収集
    for bucket_idx in 0..grid.table_size {
        let bucket = grid.buckets[bucket_idx].lock();
        let entities: Vec<Entity> = bucket.clone();
        drop(bucket);

        // 同一セル内のペア
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                collision_pairs.push((entities[i], entities[j]));
            }
        }
    }

    // 各衝突ペアに対して力を計算
    for (e1, e2) in collision_pairs {
        // 安全のためにget_manyを使用
        let Ok([p1, p2]) = particles.get_many_mut([e1, e2]) else {
            continue;
        };

        let (_, pos1, vel1, omega1, _, _, props1) = p1;
        let (_, pos2, vel2, omega2, _, _, props2) = p2;

        let contact_state = contact_history.get_or_create(e1, e2);

        let (force1, force2) = compute_particle_contact_force(
            pos1.0,
            vel1.0,
            omega1.0,
            props1.radius,
            props1.mass,
            pos2.0,
            vel2.0,
            omega2.0,
            props2.radius,
            props2.mass,
            &material,
            contact_state,
            dt,
        );

        // 力を適用（再度借用）
        if let Ok([mut p1, mut p2]) = particles.get_many_mut([e1, e2]) {
            p1.4 .0 += force1.force;
            p1.5 .0 += force1.torque;
            p2.4 .0 += force2.force;
            p2.5 .0 += force2.torque;
        }
    }
}

/// 壁との衝突を計算
pub fn compute_wall_collisions(
    container: Res<Container>,
    wall_props: Res<WallProperties>,
    mut particles: Query<(
        &Position,
        &Velocity,
        &AngularVelocity,
        &mut Force,
        &mut Torque,
        &ParticleProperties,
    )>,
) {
    particles
        .par_iter_mut()
        .for_each(|(pos, vel, omega, mut force, mut torque, props)| {
            let wall_force =
                compute_wall_contact_force(pos.0, vel.0, omega.0, props.radius, props.mass, &container, &wall_props);

            force.0 += wall_force.force;
            torque.0 += wall_force.torque;
        });
}

/// 積分の前半ステップ
pub fn integrate_positions(
    physics: Res<PhysicsConstants>,
    sim_time: Res<SimulationTime>,
    mut particles: Query<(
        &mut Position,
        &mut Velocity,
        &mut AngularVelocity,
        &Force,
        &Torque,
        &ParticleProperties,
    )>,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    particles.par_iter_mut().for_each(
        |(mut pos, mut vel, mut omega, force, torque, props)| {
            integrate_first_half(
                &mut pos.0,
                &mut vel.0,
                &mut omega.0,
                force.0,
                torque.0,
                props.mass,
                props.inertia,
                gravity,
                dt,
            );
        },
    );
}

/// 積分の後半ステップ
pub fn integrate_velocities(
    physics: Res<PhysicsConstants>,
    sim_time: Res<SimulationTime>,
    mut particles: Query<(
        &mut Velocity,
        &mut AngularVelocity,
        &Force,
        &Torque,
        &ParticleProperties,
    )>,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    particles
        .par_iter_mut()
        .for_each(|(mut vel, mut omega, force, torque, props)| {
            integrate_second_half(
                &mut vel.0,
                &mut omega.0,
                force.0,
                torque.0,
                props.mass,
                props.inertia,
                gravity,
                dt,
            );
        });
}

/// 位置と速度をクランプ（コンテナから飛び出さないように）
pub fn clamp_particles(
    container: Res<Container>,
    mut particles: Query<(
        &mut Position,
        &mut Velocity,
        &mut AngularVelocity,
        &ParticleProperties,
    )>,
) {
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;

    // 最大速度（m/s）
    const MAX_LINEAR_VEL: f32 = 10.0;
    const MAX_ANGULAR_VEL: f32 = 100.0;

    particles
        .par_iter_mut()
        .for_each(|(mut pos, mut vel, mut omega, props)| {
            // 位置をコンテナ内にクランプ
            clamp_to_container(&mut pos.0, &mut vel.0, props.radius, box_min, box_max);

            // 速度をクランプ
            clamp_velocity(&mut vel.0, &mut omega.0, MAX_LINEAR_VEL, MAX_ANGULAR_VEL);
        });
}

/// 非アクティブな接触履歴を削除
pub fn cleanup_contacts(mut contact_history: ResMut<ContactHistory>) {
    contact_history.cleanup();
}

/// 1物理ステップを実行（内部用）
fn run_single_physics_step(world: &mut World) {
    // 振動を更新
    let _ = world.run_system_once(update_oscillation_inner);
    // 空間グリッドを構築
    let _ = world.run_system_once(build_spatial_grid_inner);
    // 力をクリア
    let _ = world.run_system_once(clear_forces_inner);
    // 粒子衝突を計算
    let _ = world.run_system_once(compute_particle_collisions_inner);
    // 壁衝突を計算
    let _ = world.run_system_once(compute_wall_collisions_inner);
    // 位置を積分
    let _ = world.run_system_once(integrate_positions_inner);
    // クランプ
    let _ = world.run_system_once(clamp_particles_inner);
    // 力をクリア
    let _ = world.run_system_once(clear_forces_inner);
    // 粒子衝突を計算
    let _ = world.run_system_once(compute_particle_collisions_inner);
    // 壁衝突を計算
    let _ = world.run_system_once(compute_wall_collisions_inner);
    // 速度を積分
    let _ = world.run_system_once(integrate_velocities_inner);
    // クランプ
    let _ = world.run_system_once(clamp_particles_inner);
    // 接触履歴をクリーンアップ
    let _ = world.run_system_once(cleanup_contacts_inner);
}

// 内部システム関数（run_system_once用）
fn update_oscillation_inner(
    mut container: ResMut<Container>,
    mut params: ResMut<super::OscillationParams>,
    sim_time: Res<SimulationTime>,
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

fn build_spatial_grid_inner(
    grid: Res<SpatialHashGrid>,
    particles: Query<(Entity, &Position)>,
) {
    grid.clear();
    for (entity, pos) in particles.iter() {
        grid.insert(entity, pos.0);
    }
}

fn clear_forces_inner(mut particles: Query<(&mut Force, &mut Torque)>) {
    particles.par_iter_mut().for_each(|(mut force, mut torque)| {
        force.0 = Vec3::ZERO;
        torque.0 = Vec3::ZERO;
    });
}

fn compute_particle_collisions_inner(
    grid: Res<SpatialHashGrid>,
    mut particles: Query<(
        Entity,
        &Position,
        &Velocity,
        &AngularVelocity,
        &mut Force,
        &mut Torque,
        &ParticleProperties,
    )>,
    mut contact_history: ResMut<ContactHistory>,
    material: Res<MaterialProperties>,
    sim_time: Res<SimulationTime>,
) {
    let dt = sim_time.dt;
    let mut collision_pairs: Vec<(Entity, Entity)> = Vec::new();

    for bucket_idx in 0..grid.table_size {
        let bucket = grid.buckets[bucket_idx].lock();
        let entities: Vec<Entity> = bucket.clone();
        drop(bucket);

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                collision_pairs.push((entities[i], entities[j]));
            }
        }
    }

    for (e1, e2) in collision_pairs {
        let Ok([p1, p2]) = particles.get_many_mut([e1, e2]) else {
            continue;
        };

        let (_, pos1, vel1, omega1, _, _, props1) = p1;
        let (_, pos2, vel2, omega2, _, _, props2) = p2;

        let contact_state = contact_history.get_or_create(e1, e2);

        let (force1, force2) = compute_particle_contact_force(
            pos1.0, vel1.0, omega1.0, props1.radius, props1.mass,
            pos2.0, vel2.0, omega2.0, props2.radius, props2.mass,
            &material, contact_state, dt,
        );

        if let Ok([mut p1, mut p2]) = particles.get_many_mut([e1, e2]) {
            p1.4 .0 += force1.force;
            p1.5 .0 += force1.torque;
            p2.4 .0 += force2.force;
            p2.5 .0 += force2.torque;
        }
    }
}

fn compute_wall_collisions_inner(
    container: Res<Container>,
    wall_props: Res<WallProperties>,
    mut particles: Query<(
        &Position,
        &Velocity,
        &AngularVelocity,
        &mut Force,
        &mut Torque,
        &ParticleProperties,
    )>,
) {
    particles.par_iter_mut().for_each(|(pos, vel, omega, mut force, mut torque, props)| {
        let wall_force = compute_wall_contact_force(
            pos.0, vel.0, omega.0, props.radius, props.mass, &container, &wall_props
        );
        force.0 += wall_force.force;
        torque.0 += wall_force.torque;
    });
}

fn integrate_positions_inner(
    physics: Res<PhysicsConstants>,
    sim_time: Res<SimulationTime>,
    mut particles: Query<(
        &mut Position,
        &mut Velocity,
        &mut AngularVelocity,
        &Force,
        &Torque,
        &ParticleProperties,
    )>,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    particles.par_iter_mut().for_each(|(mut pos, mut vel, mut omega, force, torque, props)| {
        integrate_first_half(
            &mut pos.0, &mut vel.0, &mut omega.0,
            force.0, torque.0, props.mass, props.inertia, gravity, dt,
        );
    });
}

fn integrate_velocities_inner(
    physics: Res<PhysicsConstants>,
    sim_time: Res<SimulationTime>,
    mut particles: Query<(
        &mut Velocity,
        &mut AngularVelocity,
        &Force,
        &Torque,
        &ParticleProperties,
    )>,
) {
    let dt = sim_time.dt;
    let gravity = physics.gravity;

    particles.par_iter_mut().for_each(|(mut vel, mut omega, force, torque, props)| {
        integrate_second_half(
            &mut vel.0, &mut omega.0,
            force.0, torque.0, props.mass, props.inertia, gravity, dt,
        );
    });
}

fn clamp_particles_inner(
    container: Res<Container>,
    mut particles: Query<(
        &mut Position,
        &mut Velocity,
        &mut AngularVelocity,
        &ParticleProperties,
    )>,
) {
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;

    const MAX_LINEAR_VEL: f32 = 10.0;
    const MAX_ANGULAR_VEL: f32 = 100.0;

    particles.par_iter_mut().for_each(|(mut pos, mut vel, mut omega, props)| {
        clamp_to_container(&mut pos.0, &mut vel.0, props.radius, box_min, box_max);
        clamp_velocity(&mut vel.0, &mut omega.0, MAX_LINEAR_VEL, MAX_ANGULAR_VEL);
    });
}

fn cleanup_contacts_inner(mut contact_history: ResMut<ContactHistory>) {
    contact_history.cleanup();
}

/// 物理サブステップを実行するシステム
pub fn run_physics_substeps(world: &mut World) {
    // 一時停止中なら何もしない
    let paused = {
        let sim_state = world.resource::<crate::ui::SimulationState>();
        sim_state.paused
    };
    if paused {
        return;
    }

    let substeps = {
        let settings = world.resource::<SimulationSettings>();
        settings.substeps_per_frame
    };

    for _ in 0..substeps {
        run_single_physics_step(world);
        // シミュレーション時間を進める
        world.resource_mut::<SimulationTime>().step();
    }
}
