//! GPU 物理のユニットテスト

use crate::gpu::buffers::{ParticleGpu, SimulationParams};
use bevy::prelude::Vec3;
use std::collections::HashMap;

use crate::physics::compute_wall_contact_force as compute_wall_contact_force_core;
use crate::physics::{
    clamp_to_container, clamp_velocity, compute_particle_contact_force, integrate_first_half,
    integrate_second_half, ContactState, MaterialProperties, WallContactForce, WallProperties,
};
use crate::simulation::ContainerParams;

#[derive(Clone)]
struct Container {
    half_extents: Vec3,
    divider_height: f32,
    divider_thickness: f32,
    base_position: Vec3,
    current_offset: f32,
}

impl Default for Container {
    fn default() -> Self {
        let params = ContainerParams::default();
        Self {
            half_extents: params.half_extents,
            divider_height: params.divider_height,
            divider_thickness: params.divider_thickness,
            base_position: params.base_position,
            current_offset: 0.0,
        }
    }
}

impl Container {
    fn floor_y(&self) -> f32 {
        self.base_position.y - self.half_extents.y + self.current_offset
    }
}

fn compute_wall_contact_force(
    pos: Vec3,
    vel: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    container: &Container,
    wall_props: &WallProperties,
) -> WallContactForce {
    let params = ContainerParams {
        half_extents: container.half_extents,
        divider_height: container.divider_height,
        divider_thickness: container.divider_thickness,
        base_position: container.base_position,
    };
    compute_wall_contact_force_core(
        pos,
        vel,
        omega,
        radius,
        mass,
        &params,
        container.current_offset,
        wall_props,
    )
}

#[test]
fn test_particle_gpu_size_and_alignment() {
    // ParticleGpu は 64 バイト (4 x 16 bytes)
    // WGSL struct: pos(12) + radius(4) + vel(12) + mass_inv(4) + omega(12) + inertia_inv(4) + size_flag(4) + _pad(12)
    assert_eq!(
        std::mem::size_of::<ParticleGpu>(),
        64,
        "ParticleGpu should be 64 bytes"
    );
    assert_eq!(
        std::mem::align_of::<ParticleGpu>(),
        4,
        "ParticleGpu should have 4-byte alignment"
    );
}

#[test]
fn test_simulation_params_size() {
    // SimulationParams は 96 バイト
    assert_eq!(
        std::mem::size_of::<SimulationParams>(),
        96,
        "SimulationParams should be 96 bytes"
    );
}

#[test]
fn test_particle_gpu_field_offsets() {
    use std::mem::offset_of;

    // WGSL expects specific offsets
    assert_eq!(offset_of!(ParticleGpu, pos), 0);
    assert_eq!(offset_of!(ParticleGpu, radius), 12);
    assert_eq!(offset_of!(ParticleGpu, vel), 16);
    assert_eq!(offset_of!(ParticleGpu, mass_inv), 28);
    assert_eq!(offset_of!(ParticleGpu, omega), 32);
    assert_eq!(offset_of!(ParticleGpu, inertia_inv), 44);
    assert_eq!(offset_of!(ParticleGpu, size_flag), 48);
    assert_eq!(offset_of!(ParticleGpu, _pad), 52);
}

#[test]
fn test_simulation_params_field_offsets() {
    use std::mem::offset_of;

    // First 16 bytes: dt, gravity, cell_size, grid_dim
    assert_eq!(offset_of!(SimulationParams, dt), 0);
    assert_eq!(offset_of!(SimulationParams, gravity), 4);
    assert_eq!(offset_of!(SimulationParams, cell_size), 8);
    assert_eq!(offset_of!(SimulationParams, grid_dim), 12);

    // world_half (vec3<f32>) at 16, num_particles fills padding at 28
    assert_eq!(offset_of!(SimulationParams, world_half), 16);
    assert_eq!(offset_of!(SimulationParams, num_particles), 28);

    // 32-48: youngs_modulus, poisson_ratio, restitution, friction
    assert_eq!(offset_of!(SimulationParams, youngs_modulus), 32);
    assert_eq!(offset_of!(SimulationParams, poisson_ratio), 36);
    assert_eq!(offset_of!(SimulationParams, restitution), 40);
    assert_eq!(offset_of!(SimulationParams, friction), 44);

    // 48-64: container_offset, divider_height, container_half_x, container_half_y
    assert_eq!(offset_of!(SimulationParams, container_offset), 48);
    assert_eq!(offset_of!(SimulationParams, divider_height), 52);
    assert_eq!(offset_of!(SimulationParams, container_half_x), 56);
    assert_eq!(offset_of!(SimulationParams, container_half_y), 60);

    // 64-80: container_half_z, divider_thickness, rolling_friction, wall_restitution
    assert_eq!(offset_of!(SimulationParams, container_half_z), 64);
    assert_eq!(offset_of!(SimulationParams, divider_thickness), 68);
    assert_eq!(offset_of!(SimulationParams, rolling_friction), 72);
    assert_eq!(offset_of!(SimulationParams, wall_restitution), 76);
    assert_eq!(offset_of!(SimulationParams, wall_friction), 80);
    assert_eq!(offset_of!(SimulationParams, wall_damping), 84);
    assert_eq!(offset_of!(SimulationParams, wall_stiffness), 88);
    assert_eq!(offset_of!(SimulationParams, _pad_end), 92);
}

#[test]
fn test_particle_gpu_bytemuck() {
    let particle = ParticleGpu {
        pos: [1.0, 2.0, 3.0],
        radius: 0.02,
        vel: [0.1, 0.2, 0.3],
        mass_inv: 10.0,
        omega: [0.0; 3],
        inertia_inv: 1.0,
        size_flag: 1,
        _pad: [0; 3],
    };

    let bytes: &[u8] = bytemuck::bytes_of(&particle);
    assert_eq!(bytes.len(), 64);

    // Verify we can round-trip
    let particle2: ParticleGpu = *bytemuck::from_bytes(bytes);
    assert_eq!(particle.pos, particle2.pos);
    assert_eq!(particle.radius, particle2.radius);
    assert_eq!(particle.vel, particle2.vel);
    assert_eq!(particle.mass_inv, particle2.mass_inv);
    assert_eq!(particle.size_flag, particle2.size_flag);
}

#[test]
fn test_particle_slice_bytemuck() {
    let particles = vec![
        ParticleGpu {
            pos: [0.0, 0.1, 0.0],
            radius: 0.02,
            vel: [0.0, 0.0, 0.0],
            mass_inv: 10.0,
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: 1,
            _pad: [0; 3],
        },
        ParticleGpu {
            pos: [0.05, 0.1, 0.0],
            radius: 0.008,
            vel: [0.0, 0.0, 0.0],
            mass_inv: 100.0,
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: 0,
            _pad: [0; 3],
        },
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&particles);
    assert_eq!(bytes.len(), 128); // 2 * 64

    // Verify we can read back
    let particles2: &[ParticleGpu] = bytemuck::cast_slice(bytes);
    assert_eq!(particles2.len(), 2);
    assert_eq!(particles2[0].pos, particles[0].pos);
    assert_eq!(particles2[1].pos, particles[1].pos);
}

#[test]
fn test_gpu_particle_data_index_mapping() {
    use crate::gpu::plugin::GpuParticleData;
    use std::sync::Arc;

    // GpuParticleData の粒子データがインデックスで正しく対応することを検証
    let mut gpu_data = GpuParticleData::default();

    // 10個の粒子を異なるプロパティで作成
    let mut particles = Vec::new();
    for i in 0..10 {
        particles.push(ParticleGpu {
            pos: [i as f32 * 0.1, 0.0, 0.0],
            radius: if i < 5 { 0.02 } else { 0.008 },
            vel: [0.0, 0.0, 0.0],
            mass_inv: if i < 5 { 10.0 } else { 100.0 },
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: if i < 5 { 1 } else { 0 },
            _pad: [0; 3],
        });
    }
    gpu_data.particles = Arc::new(particles);

    // 粒子数が正しいことを確認
    assert_eq!(gpu_data.particles.len(), 10);

    // 各粒子が正しいデータと対応していること
    for i in 0..10 {
        assert_eq!(gpu_data.particles[i].pos[0], i as f32 * 0.1);
    }
}

#[test]
fn test_readback_data_maps_to_correct_indices() {
    use crate::gpu::readback::GpuReadbackBuffer;

    // GPU readback データが正しいインデックスにマッピングされることを検証
    let readback = GpuReadbackBuffer::default();

    // 5つの粒子を作成
    let mut particles = Vec::new();
    for i in 0..5 {
        particles.push(ParticleGpu {
            pos: [i as f32 * 0.1, 0.5, 0.0],
            radius: 0.02,
            vel: [0.0, 0.0, 0.0],
            mass_inv: 10.0,
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: 1,
            _pad: [0; 3],
        });
    }

    // GPUが物理ステップを実行した結果をシミュレート
    // （重力で Y 座標が減少、速度が増加）
    let mut simulated_results = Vec::new();
    for p in particles.iter() {
        simulated_results.push(ParticleGpu {
            pos: [p.pos[0], p.pos[1] - 0.01, p.pos[2]],
            radius: p.radius,
            vel: [0.0, -0.1, 0.0],
            mass_inv: p.mass_inv,
            omega: [0.0; 3],
            inertia_inv: p.inertia_inv,
            size_flag: p.size_flag,
            _pad: [0; 3],
        });
    }

    // readback バッファに書き込み
    {
        let mut guard = readback.data.write().unwrap();
        guard.clear();
        guard.extend_from_slice(&simulated_results);
    }
    {
        let mut frame = readback.frame.write().unwrap();
        *frame += 1;
    }

    // 読み取りと検証
    let gpu_data_read = readback.data.read().unwrap();
    assert_eq!(gpu_data_read.len(), 5);

    // 各粒子が正しいインデックスと対応していること
    for i in 0..5 {
        let p = &gpu_data_read[i];
        // X座標は元の値を保持（重力はY方向のみ）
        assert!(
            (p.pos[0] - i as f32 * 0.1).abs() < 1e-6,
            "X position mismatch at index {}: expected {}, got {}",
            i,
            i as f32 * 0.1,
            p.pos[0]
        );
        // Y座標は減少
        assert!(
            p.pos[1] < 0.5,
            "Y position should have decreased at index {}",
            i
        );
        // Y速度は負（下向き）
        assert!(
            p.vel[1] < 0.0,
            "Y velocity should be negative at index {}",
            i
        );
    }
}

#[test]
fn test_multiple_particles_distinct_after_gpu_step() {
    // GPU ステップ後、各粒子が区別可能な状態を保つことを検証
    // （全粒子が同じ値になったり、データが混ざったりしないこと）
    let num_particles = 100;
    let mut particles = Vec::with_capacity(num_particles);

    // 異なる位置に粒子を配置
    for i in 0..num_particles {
        let x = (i % 10) as f32 * 0.02 - 0.09;
        let y = (i / 10) as f32 * 0.02 + 0.05;
        let is_large = i < 20;
        particles.push(ParticleGpu {
            pos: [x, y, 0.0],
            radius: if is_large { 0.02 } else { 0.008 },
            vel: [0.0, 0.0, 0.0],
            mass_inv: if is_large { 10.0 } else { 100.0 },
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: if is_large { 1 } else { 0 },
            _pad: [0; 3],
        });
    }

    // bytemuck でシリアライズ → デシリアライズ（GPU転送をシミュレート）
    let bytes: &[u8] = bytemuck::cast_slice(&particles);
    let round_tripped: &[ParticleGpu] = bytemuck::cast_slice(bytes);

    assert_eq!(round_tripped.len(), num_particles);

    // 各粒子が元の値を保持していることを検証
    for i in 0..num_particles {
        let orig = &particles[i];
        let rt = &round_tripped[i];

        assert_eq!(orig.pos, rt.pos, "Position mismatch at particle {}", i);
        assert_eq!(orig.radius, rt.radius, "Radius mismatch at particle {}", i);
        assert_eq!(orig.vel, rt.vel, "Velocity mismatch at particle {}", i);
        assert_eq!(
            orig.mass_inv, rt.mass_inv,
            "Mass_inv mismatch at particle {}",
            i
        );
        assert_eq!(
            orig.size_flag, rt.size_flag,
            "Size flag mismatch at particle {}",
            i
        );
    }

    // 全粒子が同じ位置でないことを確認（データ混同の検出）
    let mut unique_positions = std::collections::HashSet::new();
    for p in round_tripped.iter() {
        let key = (
            (p.pos[0] * 1000.0) as i32,
            (p.pos[1] * 1000.0) as i32,
            (p.pos[2] * 1000.0) as i32,
        );
        unique_positions.insert(key);
    }
    // 100粒子で10x10グリッドなので、すべてユニークな位置
    assert_eq!(
        unique_positions.len(),
        num_particles,
        "All particles should have unique positions"
    );
}

#[test]
fn test_gpu_particle_data_generation_tracking() {
    use crate::gpu::plugin::GpuParticleData;
    use std::sync::Arc;

    // generation カウンターが正しく動作することを検証
    let mut gpu_data = GpuParticleData::default();

    assert_eq!(gpu_data.generation, 0);
    assert!(!gpu_data.initialized);

    // 初期化
    gpu_data.initialized = true;
    gpu_data.generation += 1;
    assert_eq!(gpu_data.generation, 1);

    // Reset（初期化フラグをクリア、世代をインクリメント）
    gpu_data.initialized = false;
    gpu_data.particles = Arc::new(Vec::new());

    // 再初期化
    gpu_data.initialized = true;
    gpu_data.generation += 1;
    assert_eq!(gpu_data.generation, 2);
}

/// GPU collision shader のロジックを Rust で再現（テスト用）
/// collision.wgsl の compute_contact_force と同じ計算
fn gpu_compute_contact_force(
    pos_p: [f32; 3],
    vel_p: [f32; 3],
    radius_p: f32,
    mass_inv_p: f32,
    pos_q: [f32; 3],
    vel_q: [f32; 3],
    radius_q: f32,
    mass_inv_q: f32,
    youngs_modulus: f32,
    poisson_ratio: f32,
    restitution: f32,
    friction: f32,
) -> [f32; 3] {
    let delta = [
        pos_p[0] - pos_q[0],
        pos_p[1] - pos_q[1],
        pos_p[2] - pos_q[2],
    ];
    let dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    let r_sum = radius_p + radius_q;

    if dist_sq >= r_sum * r_sum {
        return [0.0, 0.0, 0.0];
    }

    let dist = dist_sq.sqrt();
    if dist < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let mut overlap = r_sum - dist;

    if overlap <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    let n = [delta[0] / dist, delta[1] / dist, delta[2] / dist];

    let r_eff = (radius_p * radius_q) / (radius_p + radius_q);

    // overlap clamp (CPU版と同じ)
    let max_overlap = r_eff * 0.2;
    overlap = overlap.min(max_overlap);

    let e_eff = youngs_modulus / (2.0 * (1.0 - poisson_ratio * poisson_ratio));
    let k_n = (4.0 / 3.0) * e_eff * (r_eff).sqrt() * overlap.sqrt();
    let f_n_elastic = k_n * overlap;

    let rel_vel = [
        vel_p[0] - vel_q[0],
        vel_p[1] - vel_q[1],
        vel_p[2] - vel_q[2],
    ];
    let v_n = rel_vel[0] * n[0] + rel_vel[1] * n[1] + rel_vel[2] * n[2];

    let mut f_n_damping = 0.0;
    if v_n < 0.0 {
        let m_eff = 1.0 / (mass_inv_p + mass_inv_q);
        let pi = std::f32::consts::PI;
        let ln_e = restitution.ln();
        let beta = -ln_e / (pi * pi + ln_e * ln_e).sqrt();
        f_n_damping = 2.0 * beta * (k_n * m_eff).sqrt() * (-v_n);
    }

    // clamp to zero (no attraction)
    let f_n_total = (f_n_elastic + f_n_damping).max(0.0);

    // friction based on elastic only
    let v_t = [
        rel_vel[0] - v_n * n[0],
        rel_vel[1] - v_n * n[1],
        rel_vel[2] - v_n * n[2],
    ];
    let v_t_mag = (v_t[0] * v_t[0] + v_t[1] * v_t[1] + v_t[2] * v_t[2]).sqrt();

    let mut f_t = [0.0f32; 3];
    if v_t_mag > 1e-10 {
        let v_t_dir = [v_t[0] / v_t_mag, v_t[1] / v_t_mag, v_t[2] / v_t_mag];
        let f_t_max = friction * f_n_elastic;
        let v_char = 0.1;
        let viscous_coeff = f_t_max / v_char;
        let f_t_mag_clamped = (viscous_coeff * v_t_mag).min(f_t_max);
        f_t = [
            -f_t_mag_clamped * v_t_dir[0],
            -f_t_mag_clamped * v_t_dir[1],
            -f_t_mag_clamped * v_t_dir[2],
        ];
    }

    [
        f_n_total * n[0] + f_t[0],
        f_n_total * n[1] + f_t[1],
        f_n_total * n[2] + f_t[2],
    ]
}

fn kinetic_energy(vel: [f32; 3], mass_inv: f32) -> f32 {
    let mass = 1.0 / mass_inv;
    0.5 * mass * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
}

/// GPU と同じ積分ロジック（重力なし、壁なし）
fn gpu_integrate_step(
    pos: &mut [f32; 3],
    vel: &mut [f32; 3],
    force: [f32; 3],
    mass_inv: f32,
    dt: f32,
) {
    let a = [
        force[0] * mass_inv,
        force[1] * mass_inv,
        force[2] * mass_inv,
    ];
    vel[0] += a[0] * dt;
    vel[1] += a[1] * dt;
    vel[2] += a[2] * dt;
    pos[0] += vel[0] * dt;
    pos[1] += vel[1] * dt;
    pos[2] += vel[2] * dt;
}

#[test]
fn test_gpu_collision_normal_force_not_attractive() {
    // 法線力が引力（負）にならないことを検証
    // 離反速度が大きい場合でも f_n_total >= 0
    let params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // 大きな離反速度で接触中の2粒子
    // p は q の左にある → n = (p-q)/|p-q| = [-1, 0, 0]
    let pos_p = [0.0, 0.0, 0.0_f32];
    let pos_q = [0.019, 0.0, 0.0_f32]; // overlap = 0.001
    let force = gpu_compute_contact_force(
        pos_p,
        [-5.0, 0.0, 0.0], // p は左に高速離反
        0.01,
        95.493,
        pos_q,
        [5.0, 0.0, 0.0], // q は右に高速離反
        0.01,
        95.493,
        params.0,
        params.1,
        params.2,
        params.3,
    );

    // 法線方向の力が非負（引力にならない）ことを検証
    // n = (pos_p - pos_q) / dist = [-1, 0, 0]
    let delta = [
        pos_p[0] - pos_q[0],
        pos_p[1] - pos_q[1],
        pos_p[2] - pos_q[2],
    ];
    let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let n = [delta[0] / dist, delta[1] / dist, delta[2] / dist];

    // force · n = 法線方向の力成分（正 = 反発、負 = 引力）
    let f_normal = force[0] * n[0] + force[1] * n[1] + force[2] * n[2];
    assert!(
        f_normal >= 0.0,
        "Normal force component should be repulsive (non-negative), got {}",
        f_normal
    );
}

#[test]
fn test_gpu_collision_head_on_energy_conservation() {
    // 正面衝突でエネルギーが増加しないことを検証
    // 粒子は十分離れた位置からスタートし、自然に衝突させる
    let dt = 1.0 / 2000.0; // CPU テストと同じ精度
    let radius = 0.01;
    let mass_inv = 95.493; // 大粒子 (density=2500, radius=0.01)
    let params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // 十分離れた位置から接近
    let mut pos_a = [-0.05, 0.0, 0.0_f32];
    let mut vel_a = [1.0, 0.0, 0.0_f32];
    let mut pos_b = [0.05, 0.0, 0.0_f32];
    let mut vel_b = [-1.0, 0.0, 0.0_f32];

    let initial_ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);
    let mut max_ke = initial_ke;

    // 5000ステップ実行（接近→衝突→離反の全サイクル）
    for _ in 0..5000 {
        let force_on_a = gpu_compute_contact_force(
            pos_a, vel_a, radius, mass_inv, pos_b, vel_b, radius, mass_inv, params.0, params.1,
            params.2, params.3,
        );
        let force_on_b = [-force_on_a[0], -force_on_a[1], -force_on_a[2]];

        gpu_integrate_step(&mut pos_a, &mut vel_a, force_on_a, mass_inv, dt);
        gpu_integrate_step(&mut pos_b, &mut vel_b, force_on_b, mass_inv, dt);

        let ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);
        if ke > max_ke {
            max_ke = ke;
        }
    }

    let final_ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);

    // 衝突中を含めてエネルギーが増加してはいけない
    assert!(
        max_ke <= initial_ke * 1.01, // 1% の数値誤差は許容
        "Max KE during collision should not exceed initial: initial={:.6}, max={:.6}, ratio={:.4}",
        initial_ke,
        max_ke,
        max_ke / initial_ke
    );

    // 最終エネルギーも初期以下（反発係数 0.5 で減衰）
    assert!(
        final_ke <= initial_ke * 1.01,
        "Final KE should not exceed initial: initial={:.6}, final={:.6}, ratio={:.4}",
        initial_ke,
        final_ke,
        final_ke / initial_ke
    );
}

#[test]
fn test_gpu_collision_oblique_energy_conservation() {
    // 斜め衝突でもエネルギーが増加しないことを検証
    let dt = 1.0 / 2000.0;
    let radius = 0.01;
    let mass_inv = 95.493;
    let params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // 十分離れた位置からオフセットを付けて接近
    let mut pos_a = [-0.05, 0.01, 0.0_f32];
    let mut vel_a = [2.0, -0.5, 0.0_f32];
    let mut pos_b = [0.05, -0.01, 0.0_f32];
    let mut vel_b = [-2.0, 0.5, 0.0_f32];

    let initial_ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);
    let mut max_ke = initial_ke;

    for _ in 0..5000 {
        let force_on_a = gpu_compute_contact_force(
            pos_a, vel_a, radius, mass_inv, pos_b, vel_b, radius, mass_inv, params.0, params.1,
            params.2, params.3,
        );
        let force_on_b = [-force_on_a[0], -force_on_a[1], -force_on_a[2]];

        gpu_integrate_step(&mut pos_a, &mut vel_a, force_on_a, mass_inv, dt);
        gpu_integrate_step(&mut pos_b, &mut vel_b, force_on_b, mass_inv, dt);

        let ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);
        if ke > max_ke {
            max_ke = ke;
        }
    }

    let final_ke = kinetic_energy(vel_a, mass_inv) + kinetic_energy(vel_b, mass_inv);

    assert!(
        max_ke <= initial_ke * 1.01,
        "Oblique max KE should not exceed initial: initial={:.6}, max={:.6}, ratio={:.4}",
        initial_ke,
        max_ke,
        max_ke / initial_ke
    );

    assert!(
        final_ke <= initial_ke * 1.01,
        "Oblique final KE should not exceed initial: initial={:.6}, final={:.6}, ratio={:.4}",
        initial_ke,
        final_ke,
        final_ke / initial_ke
    );
}

#[test]
fn test_gpu_collision_different_sizes_energy_conservation() {
    // 大小異なるサイズの粒子間衝突でエネルギー保存を検証
    let dt = 1.0 / 2000.0;
    let radius_large = 0.01;
    let mass_inv_large = 95.493; // density=2500, radius=0.01
    let radius_small = 0.004;
    let mass_inv_small = 1492.077; // density=2500, radius=0.004
    let params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // 十分離れた位置から接近
    let mut pos_a = [-0.05, 0.0, 0.0_f32];
    let mut vel_a = [1.0, 0.0, 0.0_f32];
    let mut pos_b = [0.05, 0.0, 0.0_f32];
    let mut vel_b = [-0.5, 0.0, 0.0_f32];

    let initial_ke = kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);
    let mut max_ke = initial_ke;

    for _ in 0..5000 {
        let force_on_a = gpu_compute_contact_force(
            pos_a,
            vel_a,
            radius_large,
            mass_inv_large,
            pos_b,
            vel_b,
            radius_small,
            mass_inv_small,
            params.0,
            params.1,
            params.2,
            params.3,
        );
        let force_on_b = [-force_on_a[0], -force_on_a[1], -force_on_a[2]];

        gpu_integrate_step(&mut pos_a, &mut vel_a, force_on_a, mass_inv_large, dt);
        gpu_integrate_step(&mut pos_b, &mut vel_b, force_on_b, mass_inv_small, dt);

        let ke = kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);
        if ke > max_ke {
            max_ke = ke;
        }
    }

    let final_ke = kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);

    assert!(
        max_ke <= initial_ke * 1.01,
        "Different-size max KE should not exceed initial: initial={:.6}, max={:.6}, ratio={:.4}",
        initial_ke,
        max_ke,
        max_ke / initial_ke
    );

    assert!(
            final_ke <= initial_ke * 1.01,
            "Different-size final KE should not exceed initial: initial={:.6}, final={:.6}, ratio={:.4}",
            initial_ke, final_ke, final_ke / initial_ke
        );
}

#[test]
fn test_gpu_collision_overlap_clamp() {
    // 深い貫通でも力が異常に大きくならないことを検証
    let params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);
    let radius = 0.01;
    let mass_inv = 95.493;

    // 完全に重なった粒子（overlap = 0.02 = 2*radius）
    let force_deep = gpu_compute_contact_force(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        radius,
        mass_inv,
        [0.001, 0.0, 0.0], // overlap = 0.019
        [0.0, 0.0, 0.0],
        radius,
        mass_inv,
        params.0,
        params.1,
        params.2,
        params.3,
    );

    // 浅い貫通
    let force_shallow = gpu_compute_contact_force(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        radius,
        mass_inv,
        [0.019, 0.0, 0.0], // overlap = 0.001
        [0.0, 0.0, 0.0],
        radius,
        mass_inv,
        params.0,
        params.1,
        params.2,
        params.3,
    );

    // overlap clamp により、深い貫通の力は制限される
    // overlap clamp = r_eff * 0.2 = 0.005 * 0.2 = 0.001
    // 深い貫通 (overlap=0.019) も 0.001 にクランプされるので
    // 力は浅い貫通 (overlap=0.001) と同程度になるはず
    let force_deep_mag = (force_deep[0] * force_deep[0]
        + force_deep[1] * force_deep[1]
        + force_deep[2] * force_deep[2])
        .sqrt();
    let force_shallow_mag = (force_shallow[0] * force_shallow[0]
        + force_shallow[1] * force_shallow[1]
        + force_shallow[2] * force_shallow[2])
        .sqrt();

    // 力の大きさが近いことを確認（clamp が効いている）
    assert!(
        force_deep_mag < force_shallow_mag * 2.0,
        "Deep penetration force should be clamped: deep={:.2}, shallow={:.2}",
        force_deep_mag,
        force_shallow_mag
    );
}

/// CPU の compute_particle_contact_force を呼び出すヘルパー（角速度ゼロ）
fn cpu_compute_contact_force(
    pos_p: [f32; 3],
    vel_p: [f32; 3],
    radius_p: f32,
    mass_p: f32,
    pos_q: [f32; 3],
    vel_q: [f32; 3],
    radius_q: f32,
    mass_q: f32,
    material: &crate::physics::MaterialProperties,
) -> [f32; 3] {
    use bevy::math::Vec3;
    let mut contact_state = crate::physics::ContactState::default();
    let (force_i, _force_j) = crate::physics::compute_particle_contact_force(
        Vec3::from_array(pos_p),
        Vec3::from_array(vel_p),
        Vec3::ZERO, // angular velocity
        radius_p,
        mass_p,
        Vec3::from_array(pos_q),
        Vec3::from_array(vel_q),
        Vec3::ZERO, // angular velocity
        radius_q,
        mass_q,
        material,
        &mut contact_state,
        0.0, // dt (unused in current implementation)
    );
    force_i.force.to_array()
}

/// CPU と GPU の接触力を比較するテスト
/// 同じ条件での力の一致を検証
#[test]
fn test_cpu_gpu_contact_force_consistency() {
    let material = crate::physics::MaterialProperties {
        youngs_modulus: 1e6,
        poisson_ratio: 0.25,
        restitution: 0.5,
        friction: 0.5,
        rolling_friction: 0.1,
    };
    let gpu_params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // テストケース: (pos_p, vel_p, radius_p, mass_p, pos_q, vel_q, radius_q, mass_q, label)
    let cases: Vec<(
        [f32; 3],
        [f32; 3],
        f32,
        f32,
        [f32; 3],
        [f32; 3],
        f32,
        f32,
        &str,
    )> = vec![
        // Case 1: 同サイズ、静止、浅い接触
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            [0.019, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            "same-size stationary shallow contact",
        ),
        // Case 2: 同サイズ、接近中
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            [0.019, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            "same-size approaching",
        ),
        // Case 3: 同サイズ、離反中
        (
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            [0.019, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            "same-size separating",
        ),
        // Case 4: 異サイズ、接近中
        (
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            0.01,
            1.0 / 95.493,
            [0.013, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            0.004,
            1.0 / 1492.077,
            "different-size approaching",
        ),
        // Case 5: 斜め接触、接線速度あり
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            0.01,
            1.0 / 95.493,
            [0.019, 0.001, 0.0],
            [-1.0, -0.5, 0.0],
            0.01,
            1.0 / 95.493,
            "oblique with tangential velocity",
        ),
    ];

    for (pos_p, vel_p, radius_p, mass_p, pos_q, vel_q, radius_q, mass_q, label) in &cases {
        let cpu_force = cpu_compute_contact_force(
            *pos_p, *vel_p, *radius_p, *mass_p, *pos_q, *vel_q, *radius_q, *mass_q, &material,
        );

        let gpu_force = gpu_compute_contact_force(
            *pos_p,
            *vel_p,
            *radius_p,
            1.0 / mass_p,
            *pos_q,
            *vel_q,
            *radius_q,
            1.0 / mass_q,
            gpu_params.0,
            gpu_params.1,
            gpu_params.2,
            gpu_params.3,
        );

        let cpu_mag = (cpu_force[0].powi(2) + cpu_force[1].powi(2) + cpu_force[2].powi(2)).sqrt();
        let gpu_mag = (gpu_force[0].powi(2) + gpu_force[1].powi(2) + gpu_force[2].powi(2)).sqrt();

        // 力がゼロでなければ方向と大きさを比較
        if cpu_mag > 1e-10 && gpu_mag > 1e-10 {
            // 大きさの比率
            let mag_ratio = gpu_mag / cpu_mag;

            // 方向の一致（コサイン類似度）
            let dot = cpu_force[0] * gpu_force[0]
                + cpu_force[1] * gpu_force[1]
                + cpu_force[2] * gpu_force[2];
            let cos_sim = dot / (cpu_mag * gpu_mag);

            println!(
                    "[{}] CPU: [{:.4}, {:.4}, {:.4}] (mag={:.4}), GPU: [{:.4}, {:.4}, {:.4}] (mag={:.4}), ratio={:.4}, cos={:.6}",
                    label,
                    cpu_force[0], cpu_force[1], cpu_force[2], cpu_mag,
                    gpu_force[0], gpu_force[1], gpu_force[2], gpu_mag,
                    mag_ratio, cos_sim,
                );

            // 方向はほぼ一致するはず
            assert!(
                cos_sim > 0.99,
                "[{}] Force direction mismatch: cos_sim={:.6}",
                label,
                cos_sim
            );

            // 大きさの差異を報告（CPU の force clamp の影響がなければ一致するはず）
            assert!(
                (mag_ratio - 1.0).abs() < 0.05,
                "[{}] Force magnitude differs: CPU={:.4}, GPU={:.4}, ratio={:.4}",
                label,
                cpu_mag,
                gpu_mag,
                mag_ratio
            );
        } else {
            // 両方ゼロなら OK
            assert!(
                cpu_mag < 1e-6 && gpu_mag < 1e-6,
                "[{}] One force is zero but not the other: CPU_mag={:.6}, GPU_mag={:.6}",
                label,
                cpu_mag,
                gpu_mag
            );
        }
    }
}

/// CPU と GPU でシミュレーションを N ステップ走らせて軌跡を比較するテスト
#[test]
fn test_cpu_gpu_trajectory_consistency() {
    let dt = 1.0 / 5000.0; // runtime dt
    let radius = 0.01;
    let mass = 1.0 / 95.493_f32;
    let mass_inv = 95.493_f32;

    let material = crate::physics::MaterialProperties {
        youngs_modulus: 1e6,
        poisson_ratio: 0.25,
        restitution: 0.5,
        friction: 0.5,
        rolling_friction: 0.1,
    };
    let gpu_params = (1e6_f32, 0.25_f32, 0.5_f32, 0.5_f32);

    // 初期状態（十分離れた位置から正面衝突）
    let init_pos_a = [-0.05_f32, 0.0, 0.0];
    let init_vel_a = [1.0_f32, 0.0, 0.0];
    let init_pos_b = [0.05_f32, 0.0, 0.0];
    let init_vel_b = [-1.0_f32, 0.0, 0.0];

    // CPU シミュレーション
    let mut cpu_pos_a = init_pos_a;
    let mut cpu_vel_a = init_vel_a;
    let mut cpu_pos_b = init_pos_b;
    let mut cpu_vel_b = init_vel_b;

    // GPU シミュレーション
    let mut gpu_pos_a = init_pos_a;
    let mut gpu_vel_a = init_vel_a;
    let mut gpu_pos_b = init_pos_b;
    let mut gpu_vel_b = init_vel_b;

    let steps = 5000;
    let mut max_pos_diff = 0.0_f32;
    let mut max_vel_diff = 0.0_f32;

    for step in 0..steps {
        // CPU step
        let cpu_force = cpu_compute_contact_force(
            cpu_pos_a, cpu_vel_a, radius, mass, cpu_pos_b, cpu_vel_b, radius, mass, &material,
        );
        let cpu_force_b = [-cpu_force[0], -cpu_force[1], -cpu_force[2]];
        gpu_integrate_step(&mut cpu_pos_a, &mut cpu_vel_a, cpu_force, mass_inv, dt);
        gpu_integrate_step(&mut cpu_pos_b, &mut cpu_vel_b, cpu_force_b, mass_inv, dt);

        // GPU step
        let gpu_force = gpu_compute_contact_force(
            gpu_pos_a,
            gpu_vel_a,
            radius,
            mass_inv,
            gpu_pos_b,
            gpu_vel_b,
            radius,
            mass_inv,
            gpu_params.0,
            gpu_params.1,
            gpu_params.2,
            gpu_params.3,
        );
        let gpu_force_b = [-gpu_force[0], -gpu_force[1], -gpu_force[2]];
        gpu_integrate_step(&mut gpu_pos_a, &mut gpu_vel_a, gpu_force, mass_inv, dt);
        gpu_integrate_step(&mut gpu_pos_b, &mut gpu_vel_b, gpu_force_b, mass_inv, dt);

        // 差分を記録
        let pos_diff = ((cpu_pos_a[0] - gpu_pos_a[0]).powi(2)
            + (cpu_pos_a[1] - gpu_pos_a[1]).powi(2)
            + (cpu_pos_a[2] - gpu_pos_a[2]).powi(2))
        .sqrt();
        let vel_diff = ((cpu_vel_a[0] - gpu_vel_a[0]).powi(2)
            + (cpu_vel_a[1] - gpu_vel_a[1]).powi(2)
            + (cpu_vel_a[2] - gpu_vel_a[2]).powi(2))
        .sqrt();

        if pos_diff > max_pos_diff {
            max_pos_diff = pos_diff;
        }
        if vel_diff > max_vel_diff {
            max_vel_diff = vel_diff;
        }

        // 衝突中（接触開始付近）の差分を詳細出力
        let dist = ((cpu_pos_a[0] - cpu_pos_b[0]).powi(2)
            + (cpu_pos_a[1] - cpu_pos_b[1]).powi(2)
            + (cpu_pos_a[2] - cpu_pos_b[2]).powi(2))
        .sqrt();
        if dist < 0.025 && step % 50 == 0 {
            println!(
                "step {}: dist={:.6}, pos_diff={:.8}, vel_diff={:.8}",
                step, dist, pos_diff, vel_diff
            );
        }
    }

    println!(
        "Final: CPU pos_a=[{:.6}, {:.6}, {:.6}], vel_a=[{:.6}, {:.6}, {:.6}]",
        cpu_pos_a[0], cpu_pos_a[1], cpu_pos_a[2], cpu_vel_a[0], cpu_vel_a[1], cpu_vel_a[2],
    );
    println!(
        "Final: GPU pos_a=[{:.6}, {:.6}, {:.6}], vel_a=[{:.6}, {:.6}, {:.6}]",
        gpu_pos_a[0], gpu_pos_a[1], gpu_pos_a[2], gpu_vel_a[0], gpu_vel_a[1], gpu_vel_a[2],
    );
    println!(
        "Max pos diff: {:.10}, max vel diff: {:.10}",
        max_pos_diff, max_vel_diff
    );

    // 位置の差は粒子半径の 1% 以内であるべき
    assert!(
        max_pos_diff < radius * 0.01,
        "Position diverged too much: max_pos_diff={:.8} (threshold={:.8})",
        max_pos_diff,
        radius * 0.01
    );

    // 速度の差も小さいべき
    assert!(
        max_vel_diff < 0.1,
        "Velocity diverged too much: max_vel_diff={:.8}",
        max_vel_diff
    );
}

#[test]
fn test_readback_buffer_thread_safety() {
    use crate::gpu::readback::GpuReadbackBuffer;

    // GpuReadbackBuffer が複数スレッドから安全にアクセスできることを検証
    let readback = GpuReadbackBuffer::default();
    let readback_clone = readback.clone();

    // 別スレッドから書き込み
    let handle = std::thread::spawn(move || {
        let particles = vec![ParticleGpu {
            pos: [1.0, 2.0, 3.0],
            radius: 0.02,
            vel: [0.1, -0.5, 0.0],
            mass_inv: 10.0,
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: 1,
            _pad: [0; 3],
        }];

        {
            let mut guard = readback_clone.data.write().unwrap();
            guard.clear();
            guard.extend_from_slice(&particles);
        }
        {
            let mut frame = readback_clone.frame.write().unwrap();
            *frame = 42;
        }
    });

    handle.join().unwrap();

    // メインスレッドから読み取り
    let data = readback.data.read().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].pos, [1.0, 2.0, 3.0]);
    assert_eq!(data[0].vel, [0.1, -0.5, 0.0]);

    let frame = readback.frame.read().unwrap();
    assert_eq!(*frame, 42);
}

#[cfg(not(target_family = "wasm"))]
fn inject_shared_types(shader_src: &str) -> String {
    let types_src = include_str!("../../assets/shaders/physics_types.wgsl")
        .lines()
        .filter(|line| !line.trim_start().starts_with("#define_import_path"))
        .collect::<Vec<_>>()
        .join("\n");

    let body = shader_src
        .lines()
        .filter(|line| {
            !line
                .trim_start()
                .starts_with("#import granular_clock::physics_types::")
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!("{types_src}\n{body}")
}

#[cfg(not(target_family = "wasm"))]
fn run_neighbor_search(
    encoder: &mut wgpu::CommandEncoder,
    params_bind_group: &wgpu::BindGroup,
    particles_bind_group: &wgpu::BindGroup,
    spatial_bind_group: &wgpu::BindGroup,
    hash_pipeline: &wgpu::ComputePipeline,
    cell_pipeline: &wgpu::ComputePipeline,
    cell_ranges: &wgpu::Buffer,
    forces: &wgpu::Buffer,
    torques: &wgpu::Buffer,
    num_particles: u32,
) {
    encoder.clear_buffer(cell_ranges, 0, None);
    encoder.clear_buffer(forces, 0, None);
    encoder.clear_buffer(torques, 0, None);

    let workgroups_particles_64 = num_particles.div_ceil(64);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_test_hash"),
            timestamp_writes: None,
        });
        pass.set_pipeline(hash_pipeline);
        pass.set_bind_group(0, params_bind_group, &[]);
        pass.set_bind_group(1, particles_bind_group, &[]);
        pass.set_bind_group(2, spatial_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_test_cell_ranges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(cell_pipeline);
        pass.set_bind_group(0, params_bind_group, &[]);
        pass.set_bind_group(1, spatial_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
    }
}

#[cfg(not(target_family = "wasm"))]
fn run_neighbor_search_with_sort(
    encoder: &mut wgpu::CommandEncoder,
    params_bind_group: &wgpu::BindGroup,
    particles_bind_group: &wgpu::BindGroup,
    spatial_bind_group: &wgpu::BindGroup,
    hash_pipeline: &wgpu::ComputePipeline,
    bitonic_sort_pipeline: &wgpu::ComputePipeline,
    cell_pipeline: &wgpu::ComputePipeline,
    cell_ranges: &wgpu::Buffer,
    forces: &wgpu::Buffer,
    torques: &wgpu::Buffer,
    num_particles: u32,
    sort_count: u32,
) {
    encoder.clear_buffer(cell_ranges, 0, None);
    encoder.clear_buffer(forces, 0, None);
    encoder.clear_buffer(torques, 0, None);

    let workgroups_sort_64 = sort_count.div_ceil(64);
    let workgroups_sort_256 = sort_count.div_ceil(256);
    let workgroups_particles_64 = num_particles.div_ceil(64);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_test_hash_sorted"),
            timestamp_writes: None,
        });
        pass.set_pipeline(hash_pipeline);
        pass.set_bind_group(0, params_bind_group, &[]);
        pass.set_bind_group(1, particles_bind_group, &[]);
        pass.set_bind_group(2, spatial_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_sort_64, 1, 1);
    }

    {
        let mut k = 2u32;
        while k <= sort_count {
            let mut j = k / 2;
            while j > 0 {
                let push_constants = [j, k, sort_count];
                let push_constant_bytes = bytemuck::bytes_of(&push_constants);

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_test_bitonic_sort"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(bitonic_sort_pipeline);
                pass.set_bind_group(0, spatial_bind_group, &[]);
                pass.set_push_constants(0, push_constant_bytes);
                pass.dispatch_workgroups(workgroups_sort_256, 1, 1);
                j /= 2;
            }
            k *= 2;
        }
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_test_cell_ranges_sorted"),
            timestamp_writes: None,
        });
        pass.set_pipeline(cell_pipeline);
        pass.set_bind_group(0, params_bind_group, &[]);
        pass.set_bind_group(1, spatial_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
    }
}

#[cfg(not(target_family = "wasm"))]
fn run_collision(
    encoder: &mut wgpu::CommandEncoder,
    params_bind_group: &wgpu::BindGroup,
    particles_bind_group: &wgpu::BindGroup,
    spatial_bind_group: &wgpu::BindGroup,
    contact_bind_group: &wgpu::BindGroup,
    collision_pipeline: &wgpu::ComputePipeline,
    num_particles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gpu_test_collision"),
        timestamp_writes: None,
    });
    pass.set_pipeline(collision_pipeline);
    pass.set_bind_group(0, params_bind_group, &[]);
    pass.set_bind_group(1, particles_bind_group, &[]);
    pass.set_bind_group(2, spatial_bind_group, &[]);
    pass.set_bind_group(3, contact_bind_group, &[]);
    pass.dispatch_workgroups(num_particles.div_ceil(64), 1, 1);
}

#[cfg(not(target_family = "wasm"))]
fn run_integrate(
    encoder: &mut wgpu::CommandEncoder,
    params_bind_group: &wgpu::BindGroup,
    particles_bind_group: &wgpu::BindGroup,
    contact_bind_group: &wgpu::BindGroup,
    integrate_pipeline: &wgpu::ComputePipeline,
    num_particles: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("gpu_test_integrate"),
        timestamp_writes: None,
    });
    pass.set_pipeline(integrate_pipeline);
    pass.set_bind_group(0, params_bind_group, &[]);
    pass.set_bind_group(1, particles_bind_group, &[]);
    pass.set_bind_group(2, contact_bind_group, &[]);
    pass.dispatch_workgroups(num_particles.div_ceil(64), 1, 1);
}

#[cfg(not(target_family = "wasm"))]
struct GpuTestPipelines {
    hash_pipeline: wgpu::ComputePipeline,
    bitonic_sort_pipeline: Option<wgpu::ComputePipeline>,
    cell_pipeline: wgpu::ComputePipeline,
    collision_pipeline: wgpu::ComputePipeline,
    integrate_first_pipeline: wgpu::ComputePipeline,
    integrate_second_pipeline: wgpu::ComputePipeline,
    params_bind_group_layout: wgpu::BindGroupLayout,
    particles_bind_group_layout: wgpu::BindGroupLayout,
    spatial_bind_group_layout: wgpu::BindGroupLayout,
    contact_bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(not(target_family = "wasm"))]
#[allow(dead_code)]
struct GpuTestBuffers {
    particles_a: wgpu::Buffer,
    particles_b: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    keys: wgpu::Buffer,
    particle_ids: wgpu::Buffer,
    cell_ranges: wgpu::Buffer,
    forces: wgpu::Buffer,
    torques: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    particles_bind_group_forward: wgpu::BindGroup,
    particles_bind_group_reverse: wgpu::BindGroup,
    spatial_bind_group: wgpu::BindGroup,
    contact_bind_group: wgpu::BindGroup,
    num_particles: u32,
    sort_count: u32,
    particle_bytes_len: u64,
}

#[cfg(not(target_family = "wasm"))]
fn create_test_params_bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

#[cfg(not(target_family = "wasm"))]
fn create_test_particles_bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[cfg(not(target_family = "wasm"))]
fn create_test_spatial_bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[cfg(not(target_family = "wasm"))]
fn create_test_contact_bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[cfg(not(target_family = "wasm"))]
fn create_test_pipelines(
    device: &wgpu::Device,
    label_prefix: &str,
    with_sort: bool,
) -> GpuTestPipelines {
    let params_bind_group_layout =
        create_test_params_bind_group_layout(device, "gpu_test_params_layout");
    let particles_bind_group_layout =
        create_test_particles_bind_group_layout(device, "gpu_test_particles_layout");
    let spatial_bind_group_layout =
        create_test_spatial_bind_group_layout(device, "gpu_test_spatial_layout");
    let contact_bind_group_layout =
        create_test_contact_bind_group_layout(device, "gpu_test_contact_layout");

    let hash_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label_prefix}_hash_pipeline_layout")),
        bind_group_layouts: &[
            &params_bind_group_layout,
            &particles_bind_group_layout,
            &spatial_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });
    let cell_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label_prefix}_cell_pipeline_layout")),
        bind_group_layouts: &[&params_bind_group_layout, &spatial_bind_group_layout],
        push_constant_ranges: &[],
    });
    let collision_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label_prefix}_collision_pipeline_layout")),
            bind_group_layouts: &[
                &params_bind_group_layout,
                &particles_bind_group_layout,
                &spatial_bind_group_layout,
                &contact_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
    let integrate_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label_prefix}_integrate_pipeline_layout")),
            bind_group_layouts: &[
                &params_bind_group_layout,
                &particles_bind_group_layout,
                &contact_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

    let hash_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_hash_keys")),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(inject_shared_types(
            include_str!("../../assets/shaders/hash_keys.wgsl"),
        ))),
    });
    let cell_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_cell_ranges")),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(inject_shared_types(
            include_str!("../../assets/shaders/cell_ranges.wgsl"),
        ))),
    });
    let collision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_collision")),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(inject_shared_types(
            include_str!("../../assets/shaders/collision.wgsl"),
        ))),
    });
    let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label_prefix}_integrate")),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(inject_shared_types(
            include_str!("../../assets/shaders/integrate.wgsl"),
        ))),
    });

    let hash_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label_prefix}_hash_pipeline")),
        layout: Some(&hash_pipeline_layout),
        module: &hash_shader,
        entry_point: Some("build_keys"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bitonic_sort_pipeline = if with_sort {
        let bitonic_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label_prefix}_bitonic_pipeline_layout")),
                bind_group_layouts: &[&spatial_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..12,
                }],
            });
        let bitonic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label_prefix}_bitonic_sort")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(inject_shared_types(
                include_str!("../../assets/shaders/bitonic_sort.wgsl"),
            ))),
        });
        Some(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{label_prefix}_bitonic_pipeline")),
                layout: Some(&bitonic_pipeline_layout),
                module: &bitonic_shader,
                entry_point: Some("bitonic_sort_step"),
                compilation_options: Default::default(),
                cache: None,
            }),
        )
    } else {
        None
    };
    let cell_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label_prefix}_cell_pipeline")),
        layout: Some(&cell_pipeline_layout),
        module: &cell_shader,
        entry_point: Some("build_cell_ranges"),
        compilation_options: Default::default(),
        cache: None,
    });
    let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label_prefix}_collision_pipeline")),
        layout: Some(&collision_pipeline_layout),
        module: &collision_shader,
        entry_point: Some("collision_response"),
        compilation_options: Default::default(),
        cache: None,
    });
    let integrate_first_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label_prefix}_integrate_first")),
            layout: Some(&integrate_pipeline_layout),
            module: &integrate_shader,
            entry_point: Some("integrate_first_half"),
            compilation_options: Default::default(),
            cache: None,
        });
    let integrate_second_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label_prefix}_integrate_second")),
            layout: Some(&integrate_pipeline_layout),
            module: &integrate_shader,
            entry_point: Some("integrate_second_half"),
            compilation_options: Default::default(),
            cache: None,
        });

    GpuTestPipelines {
        hash_pipeline,
        bitonic_sort_pipeline,
        cell_pipeline,
        collision_pipeline,
        integrate_first_pipeline,
        integrate_second_pipeline,
        params_bind_group_layout,
        particles_bind_group_layout,
        spatial_bind_group_layout,
        contact_bind_group_layout,
    }
}

#[cfg(not(target_family = "wasm"))]
fn create_test_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label_prefix: &str,
    pipelines: &GpuTestPipelines,
    initial_particles: &[ParticleGpu],
    params: &SimulationParams,
    sort_count: u32,
    num_cells: u32,
) -> GpuTestBuffers {
    let num_particles = initial_particles.len() as u32;
    let particle_bytes_len = std::mem::size_of::<ParticleGpu>() as u64 * num_particles as u64;
    let particle_bytes = bytemuck::cast_slice(initial_particles);

    let particles_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_particles_a")),
        size: particle_bytes_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let particles_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_particles_b")),
        size: particle_bytes_len,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&particles_a, 0, particle_bytes);
    queue.write_buffer(&particles_b, 0, particle_bytes);

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_params")),
        size: std::mem::size_of::<SimulationParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(params));

    let keys = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_keys")),
        size: std::mem::size_of::<u32>() as u64 * sort_count as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let particle_ids = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_particle_ids")),
        size: std::mem::size_of::<u32>() as u64 * sort_count as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cell_ranges = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_cell_ranges")),
        size: std::mem::size_of::<crate::gpu::buffers::CellRange>() as u64 * num_cells as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let forces = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_forces")),
        size: std::mem::size_of::<[f32; 4]>() as u64 * num_particles as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let torques = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label_prefix}_torques")),
        size: std::mem::size_of::<[f32; 4]>() as u64 * num_particles as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label_prefix}_params_bind_group")),
        layout: &pipelines.params_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        }],
    });

    let particles_bind_group_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label_prefix}_particles_bind_group_forward")),
        layout: &pipelines.particles_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: particles_b.as_entire_binding(),
            },
        ],
    });
    let particles_bind_group_reverse = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label_prefix}_particles_bind_group_reverse")),
        layout: &pipelines.particles_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particles_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: particles_a.as_entire_binding(),
            },
        ],
    });

    let spatial_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label_prefix}_spatial_bind_group")),
        layout: &pipelines.spatial_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: keys.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: particle_ids.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cell_ranges.as_entire_binding(),
            },
        ],
    });

    let contact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label_prefix}_contact_bind_group")),
        layout: &pipelines.contact_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: forces.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: torques.as_entire_binding(),
            },
        ],
    });

    GpuTestBuffers {
        particles_a,
        particles_b,
        params_buffer,
        keys,
        particle_ids,
        cell_ranges,
        forces,
        torques,
        params_bind_group,
        particles_bind_group_forward,
        particles_bind_group_reverse,
        spatial_bind_group,
        contact_bind_group,
        num_particles,
        sort_count,
        particle_bytes_len,
    }
}

#[cfg(not(target_family = "wasm"))]
fn run_half_step(
    encoder: &mut wgpu::CommandEncoder,
    particles_bind_group: &wgpu::BindGroup,
    pipelines: &GpuTestPipelines,
    buffers: &GpuTestBuffers,
    integrate_pipeline: &wgpu::ComputePipeline,
    with_sort: bool,
) {
    if with_sort {
        run_neighbor_search_with_sort(
            encoder,
            &buffers.params_bind_group,
            particles_bind_group,
            &buffers.spatial_bind_group,
            &pipelines.hash_pipeline,
            pipelines
                .bitonic_sort_pipeline
                .as_ref()
                .expect("bitonic pipeline must exist when with_sort=true"),
            &pipelines.cell_pipeline,
            &buffers.cell_ranges,
            &buffers.forces,
            &buffers.torques,
            buffers.num_particles,
            buffers.sort_count,
        );
    } else {
        run_neighbor_search(
            encoder,
            &buffers.params_bind_group,
            particles_bind_group,
            &buffers.spatial_bind_group,
            &pipelines.hash_pipeline,
            &pipelines.cell_pipeline,
            &buffers.cell_ranges,
            &buffers.forces,
            &buffers.torques,
            buffers.num_particles,
        );
    }
    run_collision(
        encoder,
        &buffers.params_bind_group,
        particles_bind_group,
        &buffers.spatial_bind_group,
        &buffers.contact_bind_group,
        &pipelines.collision_pipeline,
        buffers.num_particles,
    );
    run_integrate(
        encoder,
        &buffers.params_bind_group,
        particles_bind_group,
        &buffers.contact_bind_group,
        integrate_pipeline,
        buffers.num_particles,
    );
}

#[cfg(not(target_family = "wasm"))]
fn run_substep_vv(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &GpuTestPipelines,
    buffers: &GpuTestBuffers,
    with_sort: bool,
) {
    run_half_step(
        encoder,
        &buffers.particles_bind_group_forward,
        pipelines,
        buffers,
        &pipelines.integrate_first_pipeline,
        with_sort,
    );
    run_half_step(
        encoder,
        &buffers.particles_bind_group_reverse,
        pipelines,
        buffers,
        &pipelines.integrate_second_pipeline,
        with_sort,
    );
}

#[cfg(not(target_family = "wasm"))]
fn create_readback_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

#[cfg(not(target_family = "wasm"))]
fn map_read_buffer_blocking(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<Vec<ParticleGpu>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {
            let mapped = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            buffer.unmap();
            Ok(result)
        }
        Ok(Err(e)) => Err(format!("map_async failed: {e:?}")),
        Err(e) => Err(format!("map_async channel failed: {e}")),
    }
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "requires native GPU adapter/device and runs real compute passes"]
fn test_gpu_e2e_matches_cpu_single_particle() {
    use crate::physics::{
        clamp_to_container, clamp_velocity, integrate_first_half, integrate_second_half,
    };
    use std::f32::consts::PI;

    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let strict_gpu = std::env::var_os("STRICT_GPU_TEST").is_some();
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(err) => {
                if strict_gpu {
                    panic!("No GPU adapter available for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping GPU E2E test (no adapter): {err:?}");
                return;
            }
        };

        let (device, queue) = match adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
        {
            Ok(pair) => pair,
            Err(err) => {
                if strict_gpu {
                    panic!("Failed to create GPU device for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping GPU E2E test (device creation failed): {err:?}");
                return;
            }
        };

        let pipelines = create_test_pipelines(&device, "gpu_test", false);

        let container = Container::default();
        let dt = 1.0 / 5000.0;
        let density = 5000.0;
        let radius: f32 = 0.01;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);
        let mass_inv = 1.0 / mass;
        let inertia_inv = 1.0 / inertia;
        let num_particles = 1u32;
        let grid_dim = 16u32;
        let num_cells = grid_dim * grid_dim * grid_dim;

        let initial_particle = ParticleGpu {
            pos: [0.02, container.floor_y() + radius + 0.015, 0.0],
            radius,
            vel: [0.4, -0.2, 0.1],
            mass_inv,
            omega: [0.0; 3],
            inertia_inv,
            size_flag: 0,
            _pad: [0; 3],
        };

        let params = SimulationParams {
            dt,
            gravity: -9.81,
            cell_size: 0.03,
            grid_dim,
            world_half: [
                container.half_extents.x,
                container.half_extents.y,
                container.half_extents.z,
            ],
            num_particles,
            youngs_modulus: 5e6,
            poisson_ratio: 0.25,
            restitution: 0.3,
            friction: 0.5,
            container_offset: container.base_position.y + container.current_offset,
            divider_height: container.divider_height,
            container_half_x: container.half_extents.x,
            container_half_y: container.half_extents.y,
            container_half_z: container.half_extents.z,
            divider_thickness: container.divider_thickness,
            rolling_friction: 0.1,
            wall_restitution: 0.3,
            wall_friction: 0.4,
            wall_damping: 20.0,
            wall_stiffness: 10000.0,
            _pad_end: 0.0,
        };

        let buffers = create_test_buffers(
            &device,
            &queue,
            "gpu_test",
            &pipelines,
            &[initial_particle],
            &params,
            num_particles.next_power_of_two(),
            num_cells,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_test_encoder"),
        });

        let substeps = 100u32;
        for _ in 0..substeps {
            // CPU と同じ順序:
            // 近傍探索 -> 衝突 -> 積分前半 -> 再探索 -> 衝突 -> 積分後半
            run_substep_vv(&mut encoder, &pipelines, &buffers, false);
        }

        let readback =
            create_readback_buffer(&device, "gpu_test_readback", buffers.particle_bytes_len);
        encoder.copy_buffer_to_buffer(
            &buffers.particles_a,
            0,
            &readback,
            0,
            buffers.particle_bytes_len,
        );

        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let gpu_particles = map_read_buffer_blocking(&device, &readback)
            .expect("GPU readback failed in ignored E2E test");
        let gpu = gpu_particles[0];

        let mut cpu_pos = Vec3::from_array(initial_particle.pos);
        let mut cpu_vel = Vec3::from_array(initial_particle.vel);
        let mut cpu_omega = Vec3::from_array(initial_particle.omega);
        let box_offset = Vec3::Y * container.current_offset;
        let box_min = container.base_position - container.half_extents + box_offset;
        let box_max = container.base_position + container.half_extents + box_offset;
        let gravity = Vec3::new(0.0, -9.81, 0.0);

        for _ in 0..substeps {
            let wall1 = compute_wall_contact_force(
                cpu_pos,
                cpu_vel,
                cpu_omega,
                radius,
                mass,
                &container,
                &crate::physics::WallProperties::default(),
            );
            integrate_first_half(
                &mut cpu_pos,
                &mut cpu_vel,
                &mut cpu_omega,
                wall1.force,
                wall1.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            clamp_to_container(&mut cpu_pos, &mut cpu_vel, radius, box_min, box_max);
            clamp_velocity(&mut cpu_vel, &mut cpu_omega, 10.0, 100.0);

            let wall2 = compute_wall_contact_force(
                cpu_pos,
                cpu_vel,
                cpu_omega,
                radius,
                mass,
                &container,
                &crate::physics::WallProperties::default(),
            );
            integrate_second_half(
                &mut cpu_vel,
                &mut cpu_omega,
                wall2.force,
                wall2.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            clamp_to_container(&mut cpu_pos, &mut cpu_vel, radius, box_min, box_max);
            clamp_velocity(&mut cpu_vel, &mut cpu_omega, 10.0, 100.0);
        }

        let gpu_pos = Vec3::from_array(gpu.pos);
        let gpu_vel = Vec3::from_array(gpu.vel);
        let pos_diff = (cpu_pos - gpu_pos).length();
        let vel_diff = (cpu_vel - gpu_vel).length();

        assert!(
            pos_diff < 1e-3,
            "GPU E2E position mismatch too large: {pos_diff}, cpu={cpu_pos:?}, gpu={gpu_pos:?}"
        );
        assert!(
            vel_diff < 1e-2,
            "GPU E2E velocity mismatch too large: {vel_diff}, cpu={cpu_vel:?}, gpu={gpu_vel:?}"
        );
    });
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "requires native GPU adapter/device and runs real compute passes"]
fn test_gpu_e2e_matches_cpu_dense_contacts() {
    use std::f32::consts::PI;

    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let strict_gpu = std::env::var_os("STRICT_GPU_TEST").is_some();

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(err) => {
                if strict_gpu {
                    panic!("No GPU adapter available for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping GPU E2E dense test (no adapter): {err:?}");
                return;
            }
        };

        if !adapter.features().contains(wgpu::Features::PUSH_CONSTANTS) {
            if strict_gpu {
                panic!("Adapter does not support PUSH_CONSTANTS required by bitonic sort test");
            }
            eprintln!("Skipping GPU E2E dense test (PUSH_CONSTANTS unsupported)");
            return;
        }
        if adapter.limits().max_push_constant_size < 12 {
            if strict_gpu {
                panic!(
                    "Adapter max_push_constant_size={} is too small for bitonic sort",
                    adapter.limits().max_push_constant_size
                );
            }
            eprintln!(
                "Skipping GPU E2E dense test (insufficient push constants: {})",
                adapter.limits().max_push_constant_size
            );
            return;
        }

        let mut device_desc = wgpu::DeviceDescriptor::default();
        device_desc.required_features = wgpu::Features::PUSH_CONSTANTS;
        device_desc.required_limits.max_push_constant_size = 12;

        let (device, queue) = match adapter.request_device(&device_desc).await {
            Ok(pair) => pair,
            Err(err) => {
                if strict_gpu {
                    panic!("Failed to create GPU device for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping GPU E2E dense test (device creation failed): {err:?}");
                return;
            }
        };

        let pipelines = create_test_pipelines(&device, "gpu_dense_test", true);

        let container = Container::default();
        let material = MaterialProperties::default();
        let wall_props = WallProperties::default();
        let dt = 1.0 / 5000.0;
        let density = 5000.0;

        let mut initial = Vec::<BackendParticle>::new();
        for ix in 0..6 {
            for iz in 0..4 {
                let radius: f32 = if (ix + iz) % 3 == 0 { 0.01 } else { 0.006 };
                let volume = (4.0 / 3.0) * PI * radius.powi(3);
                let mass = density * volume;
                let inertia = (2.0 / 5.0) * mass * radius.powi(2);
                let pos = Vec3::new(
                    -0.12 + ix as f32 * 0.045,
                    -0.08 + (iz % 2) as f32 * 0.03,
                    -0.045 + iz as f32 * 0.03,
                );
                let vel = Vec3::new(
                    if ix % 2 == 0 { 1.0 } else { -0.95 },
                    -0.3 + iz as f32 * 0.12,
                    if iz % 2 == 0 { 0.6 } else { -0.55 },
                );
                initial.push(BackendParticle {
                    pos,
                    vel,
                    omega: Vec3::ZERO,
                    radius,
                    mass,
                    inertia,
                });
            }
        }
        for k in 0..8 {
            let radius: f32 = if k % 2 == 0 { 0.01 } else { 0.006 };
            let volume = (4.0 / 3.0) * PI * radius.powi(3);
            let mass = density * volume;
            let inertia = (2.0 / 5.0) * mass * radius.powi(2);
            let left = k < 4;
            let lane = (k % 4) as f32;
            let pos = Vec3::new(
                if left { -0.18 } else { 0.18 },
                -0.12 + lane * 0.05,
                -0.04 + lane * 0.025,
            );
            let vel = Vec3::new(
                if left { 1.4 } else { -1.4 },
                if k % 3 == 0 { 0.4 } else { -0.2 },
                if k % 2 == 0 { 0.5 } else { -0.5 },
            );
            initial.push(BackendParticle {
                pos,
                vel,
                omega: Vec3::ZERO,
                radius,
                mass,
                inertia,
            });
        }

        let num_particles = initial.len() as u32;
        let sort_count = num_particles.next_power_of_two();
        let grid_dim = 32u32;
        let num_cells = grid_dim * grid_dim * grid_dim;
        let particle_bytes_len = std::mem::size_of::<ParticleGpu>() as u64 * num_particles as u64;

        let initial_gpu: Vec<ParticleGpu> = initial
            .iter()
            .map(|p| ParticleGpu {
                pos: p.pos.to_array(),
                radius: p.radius,
                vel: p.vel.to_array(),
                mass_inv: 1.0 / p.mass,
                omega: p.omega.to_array(),
                inertia_inv: 1.0 / p.inertia,
                size_flag: if p.radius > 0.008 { 1 } else { 0 },
                _pad: [0; 3],
            })
            .collect();

        let params = SimulationParams {
            dt,
            gravity: -9.81,
            cell_size: 0.03,
            grid_dim,
            world_half: [
                container.half_extents.x,
                container.half_extents.y,
                container.half_extents.z,
            ],
            num_particles,
            youngs_modulus: material.youngs_modulus,
            poisson_ratio: material.poisson_ratio,
            restitution: material.restitution,
            friction: material.friction,
            container_offset: container.base_position.y + container.current_offset,
            divider_height: container.divider_height,
            container_half_x: container.half_extents.x,
            container_half_y: container.half_extents.y,
            container_half_z: container.half_extents.z,
            divider_thickness: container.divider_thickness,
            rolling_friction: material.rolling_friction,
            wall_restitution: wall_props.restitution,
            wall_friction: wall_props.friction,
            wall_damping: wall_props.damping,
            wall_stiffness: wall_props.stiffness,
            _pad_end: 0.0,
        };

        let buffers = create_test_buffers(
            &device,
            &queue,
            "gpu_dense_test",
            &pipelines,
            &initial_gpu,
            &params,
            sort_count,
            num_cells,
        );

        let substeps = 600u32;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_dense_test_encoder"),
        });
        for _ in 0..substeps {
            // CPU と同じ順序:
            // 近傍探索 -> 衝突 -> 積分前半 -> 再探索 -> 衝突 -> 積分後半
            run_substep_vv(&mut encoder, &pipelines, &buffers, true);
        }

        let readback =
            create_readback_buffer(&device, "gpu_dense_test_readback", particle_bytes_len);
        encoder.copy_buffer_to_buffer(&buffers.particles_a, 0, &readback, 0, particle_bytes_len);
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let gpu_particles = map_read_buffer_blocking(&device, &readback)
            .expect("GPU readback failed in ignored dense E2E test");
        assert_eq!(
            gpu_particles.len(),
            num_particles as usize,
            "GPU readback length mismatch"
        );

        let mut cpu_particles = initial.clone();
        let mut contact_states: HashMap<(usize, usize), ContactState> = HashMap::new();
        let mut particle_contacts_total = 0usize;
        let mut wall_contacts_total = 0usize;

        let box_offset = Vec3::Y * container.current_offset;
        let box_min = container.base_position - container.half_extents + box_offset;
        let box_max = container.base_position + container.half_extents + box_offset;
        let divider_top = box_min.y + container.divider_height;
        let divider_half_thickness = container.divider_thickness * 0.5;

        let count_particle_overlaps = |particles: &[BackendParticle]| -> usize {
            let mut count = 0usize;
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let p = &particles[i];
                    let q = &particles[j];
                    if (p.pos - q.pos).length() < (p.radius + q.radius) {
                        count += 1;
                    }
                }
            }
            count
        };
        let count_wall_overlaps = |particles: &[BackendParticle]| -> usize {
            let mut count = 0usize;
            for p in particles {
                let wall_hit = p.pos.y - p.radius <= box_min.y
                    || p.pos.y + p.radius >= box_max.y
                    || p.pos.x - p.radius <= box_min.x
                    || p.pos.x + p.radius >= box_max.x
                    || p.pos.z - p.radius <= box_min.z
                    || p.pos.z + p.radius >= box_max.z;
                let divider_hit = (p.pos.y - p.radius) < divider_top
                    && (p.pos.x.abs() - divider_half_thickness) < p.radius;
                if wall_hit || divider_hit {
                    count += 1;
                }
            }
            count
        };

        for _ in 0..substeps {
            particle_contacts_total += count_particle_overlaps(&cpu_particles);
            wall_contacts_total += count_wall_overlaps(&cpu_particles);
            cpu_substep(
                &mut cpu_particles,
                &mut contact_states,
                &container,
                &material,
                &wall_props,
                dt,
            );
        }

        assert!(
            particle_contacts_total > 100,
            "Scenario did not produce enough particle-particle contacts: {}",
            particle_contacts_total
        );
        assert!(
            wall_contacts_total > 100,
            "Scenario did not produce enough wall contacts: {}",
            wall_contacts_total
        );

        let mut max_pos_diff = 0.0f32;
        let mut max_vel_diff = 0.0f32;
        let mut sum_pos_diff = 0.0f32;
        let mut sum_vel_diff = 0.0f32;

        for i in 0..num_particles as usize {
            let gpu = &gpu_particles[i];
            let gpu_pos = Vec3::from_array(gpu.pos);
            let gpu_vel = Vec3::from_array(gpu.vel);

            let pos_diff = (cpu_particles[i].pos - gpu_pos).length();
            let vel_diff = (cpu_particles[i].vel - gpu_vel).length();
            max_pos_diff = max_pos_diff.max(pos_diff);
            max_vel_diff = max_vel_diff.max(vel_diff);
            sum_pos_diff += pos_diff;
            sum_vel_diff += vel_diff;
        }

        let mean_pos_diff = sum_pos_diff / num_particles as f32;
        let mean_vel_diff = sum_vel_diff / num_particles as f32;
        println!(
            "Dense E2E: contacts particle={}, wall={}, max_pos_diff={:.6}, mean_pos_diff={:.6}, max_vel_diff={:.6}, mean_vel_diff={:.6}",
            particle_contacts_total,
            wall_contacts_total,
            max_pos_diff,
            mean_pos_diff,
            max_vel_diff,
            mean_vel_diff
        );

        assert!(
            mean_pos_diff < 1.5e-2,
            "Dense E2E mean position mismatch too large: {}",
            mean_pos_diff
        );
        assert!(
            max_pos_diff < 6.0e-2,
            "Dense E2E max position mismatch too large: {}",
            max_pos_diff
        );
        assert!(
            mean_vel_diff < 1.5e-1,
            "Dense E2E mean velocity mismatch too large: {}",
            mean_vel_diff
        );
        assert!(
            max_vel_diff < 7.0e-1,
            "Dense E2E max velocity mismatch too large: {}",
            max_vel_diff
        );
    });
}

#[cfg(not(target_family = "wasm"))]
#[test]
#[ignore = "requires native GPU adapter/device and runs real compute passes"]
fn test_gpu_e2e_matches_cpu_compact_vibrating_box() {
    use std::f32::consts::PI;

    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let strict_gpu = std::env::var_os("STRICT_GPU_TEST").is_some();

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(err) => {
                if strict_gpu {
                    panic!("No GPU adapter available for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping compact vibrating E2E test (no adapter): {err:?}");
                return;
            }
        };

        if !adapter.features().contains(wgpu::Features::PUSH_CONSTANTS) {
            if strict_gpu {
                panic!("Adapter does not support PUSH_CONSTANTS required by bitonic sort test");
            }
            eprintln!("Skipping compact vibrating E2E test (PUSH_CONSTANTS unsupported)");
            return;
        }
        if adapter.limits().max_push_constant_size < 12 {
            if strict_gpu {
                panic!(
                    "Adapter max_push_constant_size={} is too small for bitonic sort",
                    adapter.limits().max_push_constant_size
                );
            }
            eprintln!(
                "Skipping compact vibrating E2E test (insufficient push constants: {})",
                adapter.limits().max_push_constant_size
            );
            return;
        }

        let mut device_desc = wgpu::DeviceDescriptor::default();
        device_desc.required_features = wgpu::Features::PUSH_CONSTANTS;
        device_desc.required_limits.max_push_constant_size = 12;

        let (device, queue) = match adapter.request_device(&device_desc).await {
            Ok(pair) => pair,
            Err(err) => {
                if strict_gpu {
                    panic!("Failed to create GPU device for ignored GPU E2E test: {err:?}");
                }
                eprintln!("Skipping compact vibrating E2E test (device creation failed): {err:?}");
                return;
            }
        };

        let pipelines = create_test_pipelines(&device, "gpu_compact_vib_test", true);

        let mut container = Container::default();
        container.half_extents = Vec3::new(0.06, 0.06, 0.04);
        container.base_position = Vec3::new(0.0, 0.0, 0.0);
        container.divider_height = 0.04;
        container.divider_thickness = 0.01;
        container.current_offset = 0.0;

        let material = MaterialProperties::default();
        let wall_props = WallProperties::default();
        let dt = 1.0 / 5000.0;
        let density = 5000.0;
        let oscillation_amplitude = 0.006;
        let oscillation_frequency = 22.0;

        let mut initial = Vec::<BackendParticle>::new();
        for ix in 0..5 {
            for iy in 0..4 {
                for iz in 0..3 {
                    let radius: f32 = if (ix + iy + iz) % 2 == 0 {
                        0.0055
                    } else {
                        0.0048
                    };
                    let volume = (4.0 / 3.0) * PI * radius.powi(3);
                    let mass = density * volume;
                    let inertia = (2.0 / 5.0) * mass * radius.powi(2);
                    let pos = Vec3::new(
                        -0.036 + ix as f32 * 0.018,
                        -0.036 + iy as f32 * 0.016,
                        -0.018 + iz as f32 * 0.018,
                    );
                    let vel = Vec3::new(
                        if (ix + iy) % 2 == 0 { 0.35 } else { -0.32 },
                        if iy % 2 == 0 { 0.08 } else { -0.06 },
                        if iz % 2 == 0 { 0.28 } else { -0.26 },
                    );
                    initial.push(BackendParticle {
                        pos,
                        vel,
                        omega: Vec3::ZERO,
                        radius,
                        mass,
                        inertia,
                    });
                }
            }
        }

        let num_particles = initial.len() as u32;
        let sort_count = num_particles.next_power_of_two();
        let grid_dim = 24u32;
        let num_cells = grid_dim * grid_dim * grid_dim;
        let particle_bytes_len = std::mem::size_of::<ParticleGpu>() as u64 * num_particles as u64;

        let initial_gpu: Vec<ParticleGpu> = initial
            .iter()
            .map(|p| ParticleGpu {
                pos: p.pos.to_array(),
                radius: p.radius,
                vel: p.vel.to_array(),
                mass_inv: 1.0 / p.mass,
                omega: p.omega.to_array(),
                inertia_inv: 1.0 / p.inertia,
                size_flag: if p.radius > 0.005 { 1 } else { 0 },
                _pad: [0; 3],
            })
            .collect();

        let mut params = SimulationParams {
            dt,
            gravity: -9.81,
            cell_size: 0.012,
            grid_dim,
            world_half: [
                container.half_extents.x,
                container.half_extents.y,
                container.half_extents.z,
            ],
            num_particles,
            youngs_modulus: material.youngs_modulus,
            poisson_ratio: material.poisson_ratio,
            restitution: material.restitution,
            friction: material.friction,
            container_offset: container.base_position.y + container.current_offset,
            divider_height: container.divider_height,
            container_half_x: container.half_extents.x,
            container_half_y: container.half_extents.y,
            container_half_z: container.half_extents.z,
            divider_thickness: container.divider_thickness,
            rolling_friction: material.rolling_friction,
            wall_restitution: wall_props.restitution,
            wall_friction: wall_props.friction,
            wall_damping: wall_props.damping,
            wall_stiffness: wall_props.stiffness,
            _pad_end: 0.0,
        };

        let buffers = create_test_buffers(
            &device,
            &queue,
            "gpu_compact_vib_test",
            &pipelines,
            &initial_gpu,
            &params,
            sort_count,
            num_cells,
        );

        let substeps = 320u32;
        let offset_for_step = |step: u32| -> f32 {
            let t = (step as f32 + 1.0) * dt;
            oscillation_amplitude * (2.0 * PI * oscillation_frequency * t).sin()
        };

        for step in 0..substeps {
            params.container_offset = container.base_position.y + offset_for_step(step);
            queue.write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_compact_vib_test_encoder"),
            });
            // CPU と同じ順序:
            // 近傍探索 -> 衝突 -> 積分前半 -> 再探索 -> 衝突 -> 積分後半
            run_substep_vv(&mut encoder, &pipelines, &buffers, true);
            queue.submit([encoder.finish()]);
        }

        let readback =
            create_readback_buffer(&device, "gpu_compact_vib_test_readback", particle_bytes_len);
        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu_compact_vib_test_copy_encoder"),
        });
        copy_encoder.copy_buffer_to_buffer(
            &buffers.particles_a,
            0,
            &readback,
            0,
            particle_bytes_len,
        );
        queue.submit([copy_encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let gpu_particles = map_read_buffer_blocking(&device, &readback)
            .expect("GPU readback failed in ignored compact vibrating E2E test");
        assert_eq!(
            gpu_particles.len(),
            num_particles as usize,
            "GPU readback length mismatch"
        );

        let mut cpu_container = container.clone();
        let mut cpu_particles = initial.clone();
        let mut contact_states: HashMap<(usize, usize), ContactState> = HashMap::new();
        let mut particle_contacts_total = 0usize;
        let mut wall_contacts_total = 0usize;

        let count_particle_overlaps = |particles: &[BackendParticle]| -> usize {
            let mut count = 0usize;
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let p = &particles[i];
                    let q = &particles[j];
                    if (p.pos - q.pos).length() < (p.radius + q.radius) {
                        count += 1;
                    }
                }
            }
            count
        };

        let count_wall_overlaps = |particles: &[BackendParticle], c: &Container| -> usize {
            let box_offset = Vec3::Y * c.current_offset;
            let box_min = c.base_position - c.half_extents + box_offset;
            let box_max = c.base_position + c.half_extents + box_offset;
            let divider_top = box_min.y + c.divider_height;
            let divider_half_thickness = c.divider_thickness * 0.5;

            let mut count = 0usize;
            for p in particles {
                let wall_hit = p.pos.y - p.radius <= box_min.y
                    || p.pos.y + p.radius >= box_max.y
                    || p.pos.x - p.radius <= box_min.x
                    || p.pos.x + p.radius >= box_max.x
                    || p.pos.z - p.radius <= box_min.z
                    || p.pos.z + p.radius >= box_max.z;
                let divider_hit = (p.pos.y - p.radius) < divider_top
                    && (p.pos.x.abs() - divider_half_thickness) < p.radius;
                if wall_hit || divider_hit {
                    count += 1;
                }
            }
            count
        };

        for step in 0..substeps {
            cpu_container.current_offset = offset_for_step(step);
            particle_contacts_total += count_particle_overlaps(&cpu_particles);
            wall_contacts_total += count_wall_overlaps(&cpu_particles, &cpu_container);
            cpu_substep(
                &mut cpu_particles,
                &mut contact_states,
                &cpu_container,
                &material,
                &wall_props,
                dt,
            );
        }

        assert!(
            particle_contacts_total > 400,
            "Compact vibrating scenario did not produce enough particle contacts: {}",
            particle_contacts_total
        );
        assert!(
            wall_contacts_total > 400,
            "Compact vibrating scenario did not produce enough wall contacts: {}",
            wall_contacts_total
        );

        let mut max_pos_diff = 0.0f32;
        let mut max_vel_diff = 0.0f32;
        let mut sum_pos_diff = 0.0f32;
        let mut sum_vel_diff = 0.0f32;
        for i in 0..num_particles as usize {
            let gpu = &gpu_particles[i];
            let gpu_pos = Vec3::from_array(gpu.pos);
            let gpu_vel = Vec3::from_array(gpu.vel);

            let pos_diff = (cpu_particles[i].pos - gpu_pos).length();
            let vel_diff = (cpu_particles[i].vel - gpu_vel).length();
            max_pos_diff = max_pos_diff.max(pos_diff);
            max_vel_diff = max_vel_diff.max(vel_diff);
            sum_pos_diff += pos_diff;
            sum_vel_diff += vel_diff;
        }

        let mean_pos_diff = sum_pos_diff / num_particles as f32;
        let mean_vel_diff = sum_vel_diff / num_particles as f32;
        println!(
            "Compact vibrating E2E: contacts particle={}, wall={}, max_pos_diff={:.6}, mean_pos_diff={:.6}, max_vel_diff={:.6}, mean_vel_diff={:.6}",
            particle_contacts_total,
            wall_contacts_total,
            max_pos_diff,
            mean_pos_diff,
            max_vel_diff,
            mean_vel_diff
        );

        assert!(
            mean_pos_diff < 2.5e-2,
            "Compact vibrating E2E mean position mismatch too large: {}",
            mean_pos_diff
        );
        assert!(
            max_pos_diff < 9.0e-2,
            "Compact vibrating E2E max position mismatch too large: {}",
            max_pos_diff
        );
        assert!(
            mean_vel_diff < 2.5e-1,
            "Compact vibrating E2E mean velocity mismatch too large: {}",
            mean_vel_diff
        );
        assert!(
            max_vel_diff < 1.2,
            "Compact vibrating E2E max velocity mismatch too large: {}",
            max_vel_diff
        );
    });
}

#[derive(Clone)]
struct BackendParticle {
    pos: Vec3,
    vel: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    inertia: f32,
}

fn cpu_substep(
    particles: &mut [BackendParticle],
    contact_states: &mut HashMap<(usize, usize), ContactState>,
    container: &Container,
    material: &MaterialProperties,
    wall_props: &WallProperties,
    dt: f32,
) {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;
    let n = particles.len();

    let mut forces = vec![Vec3::ZERO; n];
    let mut torques = vec![Vec3::ZERO; n];

    for i in 0..n {
        let p = &particles[i];
        let wall = compute_wall_contact_force(
            p.pos, p.vel, p.omega, p.radius, p.mass, container, wall_props,
        );
        forces[i] += wall.force;
        torques[i] += wall.torque;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let p_i = &particles[i];
            let p_j = &particles[j];
            let state = contact_states.entry((i, j)).or_default();
            let (f_i, f_j) = compute_particle_contact_force(
                p_i.pos, p_i.vel, p_i.omega, p_i.radius, p_i.mass, p_j.pos, p_j.vel, p_j.omega,
                p_j.radius, p_j.mass, material, state, dt,
            );
            forces[i] += f_i.force;
            torques[i] += f_i.torque;
            forces[j] += f_j.force;
            torques[j] += f_j.torque;
        }
    }

    for i in 0..n {
        let p = &mut particles[i];
        integrate_first_half(
            &mut p.pos,
            &mut p.vel,
            &mut p.omega,
            forces[i],
            torques[i],
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
        clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
        clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
    }

    forces.fill(Vec3::ZERO);
    torques.fill(Vec3::ZERO);

    for i in 0..n {
        let p = &particles[i];
        let wall = compute_wall_contact_force(
            p.pos, p.vel, p.omega, p.radius, p.mass, container, wall_props,
        );
        forces[i] += wall.force;
        torques[i] += wall.torque;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let p_i = &particles[i];
            let p_j = &particles[j];
            let state = contact_states.entry((i, j)).or_default();
            let (f_i, f_j) = compute_particle_contact_force(
                p_i.pos, p_i.vel, p_i.omega, p_i.radius, p_i.mass, p_j.pos, p_j.vel, p_j.omega,
                p_j.radius, p_j.mass, material, state, dt,
            );
            forces[i] += f_i.force;
            torques[i] += f_i.torque;
            forces[j] += f_j.force;
            torques[j] += f_j.torque;
        }
    }

    for i in 0..n {
        let p = &mut particles[i];
        integrate_second_half(
            &mut p.vel,
            &mut p.omega,
            forces[i],
            torques[i],
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
        clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
        clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
    }

    // ContactHistory::cleanup 相当（現在のモデルでは last_normal 以外には影響しない）
    contact_states.retain(|_, state| state.active);
    for state in contact_states.values_mut() {
        state.active = false;
    }
}

fn gpu_sequence_substep(
    particles: &mut [BackendParticle],
    container: &Container,
    material: &MaterialProperties,
    wall_props: &WallProperties,
    dt: f32,
) {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;
    let n = particles.len();

    let mut forces = vec![Vec3::ZERO; n];
    let mut torques = vec![Vec3::ZERO; n];
    let snapshot = particles.to_vec();

    for i in 0..n {
        let p = &snapshot[i];
        let wall = compute_wall_contact_force(
            p.pos, p.vel, p.omega, p.radius, p.mass, container, wall_props,
        );
        forces[i] += wall.force;
        torques[i] += wall.torque;

        for (j, q) in snapshot.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut tmp = ContactState::default();
            let (f_i, _) = compute_particle_contact_force(
                p.pos, p.vel, p.omega, p.radius, p.mass, q.pos, q.vel, q.omega, q.radius, q.mass,
                material, &mut tmp, dt,
            );
            forces[i] += f_i.force;
            torques[i] += f_i.torque;
        }
    }

    for i in 0..n {
        let p = &mut particles[i];
        integrate_first_half(
            &mut p.pos,
            &mut p.vel,
            &mut p.omega,
            forces[i],
            torques[i],
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
        clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
        clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
    }

    forces.fill(Vec3::ZERO);
    torques.fill(Vec3::ZERO);
    let snapshot = particles.to_vec();

    for i in 0..n {
        let p = &snapshot[i];
        let wall = compute_wall_contact_force(
            p.pos, p.vel, p.omega, p.radius, p.mass, container, wall_props,
        );
        forces[i] += wall.force;
        torques[i] += wall.torque;

        for (j, q) in snapshot.iter().enumerate() {
            if j == i {
                continue;
            }
            let mut tmp = ContactState::default();
            let (f_i, _) = compute_particle_contact_force(
                p.pos, p.vel, p.omega, p.radius, p.mass, q.pos, q.vel, q.omega, q.radius, q.mass,
                material, &mut tmp, dt,
            );
            forces[i] += f_i.force;
            torques[i] += f_i.torque;
        }
    }

    for i in 0..n {
        let p = &mut particles[i];
        integrate_second_half(
            &mut p.vel,
            &mut p.omega,
            forces[i],
            torques[i],
            p.mass,
            p.inertia,
            gravity,
            dt,
        );
        clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
        clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
    }
}

#[test]
fn test_cpu_gpu_sequence_parity_with_walls() {
    use std::f32::consts::PI;

    let container = Container::default();
    let material = MaterialProperties::default();
    let wall_props = WallProperties::default();
    let dt = 1.0 / 5000.0;
    let density = 5000.0;

    let mk_particle = |pos: Vec3, vel: Vec3, radius: f32| {
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);
        BackendParticle {
            pos,
            vel,
            omega: Vec3::ZERO,
            radius,
            mass,
            inertia,
        }
    };

    let initial = vec![
        mk_particle(
            Vec3::new(-0.08, -0.02, 0.00),
            Vec3::new(0.8, -0.4, 0.2),
            0.01,
        ),
        mk_particle(
            Vec3::new(-0.05, -0.01, 0.01),
            Vec3::new(-0.3, -0.2, 0.1),
            0.006,
        ),
        mk_particle(
            Vec3::new(-0.02, 0.00, -0.01),
            Vec3::new(0.4, -0.1, -0.3),
            0.006,
        ),
        mk_particle(Vec3::new(0.03, 0.01, 0.00), Vec3::new(-0.5, 0.0, 0.2), 0.01),
        mk_particle(
            Vec3::new(0.06, 0.02, -0.01),
            Vec3::new(0.1, -0.3, 0.4),
            0.006,
        ),
        mk_particle(
            Vec3::new(0.09, 0.03, 0.01),
            Vec3::new(-0.2, -0.2, -0.2),
            0.006,
        ),
    ];

    let mut cpu_particles = initial.clone();
    let mut gpu_particles = initial;
    let mut contact_states: HashMap<(usize, usize), ContactState> = HashMap::new();

    for _ in 0..300 {
        cpu_substep(
            &mut cpu_particles,
            &mut contact_states,
            &container,
            &material,
            &wall_props,
            dt,
        );
        gpu_sequence_substep(&mut gpu_particles, &container, &material, &wall_props, dt);
    }

    let mut max_pos_diff = 0.0f32;
    let mut max_vel_diff = 0.0f32;
    for i in 0..cpu_particles.len() {
        let pos_diff = (cpu_particles[i].pos - gpu_particles[i].pos).length();
        let vel_diff = (cpu_particles[i].vel - gpu_particles[i].vel).length();
        max_pos_diff = max_pos_diff.max(pos_diff);
        max_vel_diff = max_vel_diff.max(vel_diff);
    }

    assert!(
        max_pos_diff < 2e-3,
        "CPU/GPU sequence position mismatch too large: {}",
        max_pos_diff
    );
    assert!(
        max_vel_diff < 2e-2,
        "CPU/GPU sequence velocity mismatch too large: {}",
        max_vel_diff
    );
}
