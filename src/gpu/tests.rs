//! GPU 物理のユニットテスト

#[cfg(test)]
mod tests {
    use crate::gpu::buffers::{ParticleGpu, SimulationParams};

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
        // SimulationParams は 80 バイト
        // WGSL: vec3<f32> は 12 バイトだが、その後に u32 が続くので 28 からスタート可能
        assert_eq!(
            std::mem::size_of::<SimulationParams>(),
            80,
            "SimulationParams should be 80 bytes"
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

        // 64-80: container_half_z, divider_thickness, rolling_friction, _pad
        assert_eq!(offset_of!(SimulationParams, container_half_z), 64);
        assert_eq!(offset_of!(SimulationParams, divider_thickness), 68);
        assert_eq!(offset_of!(SimulationParams, rolling_friction), 72);
        assert_eq!(offset_of!(SimulationParams, _pad), 76);
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
    fn test_gpu_particle_data_entity_mapping() {
        use crate::gpu::plugin::GpuParticleData;
        use bevy::prelude::*;

        // GpuParticleData のエンティティと粒子データの対応が正しいことを検証
        let mut gpu_data = GpuParticleData::default();

        // 10個の粒子を異なるプロパティで作成
        let mut world = World::new();
        let mut entities = Vec::new();
        for i in 0..10 {
            let entity = world.spawn_empty().id();
            entities.push(entity);
            gpu_data.entities.push(entity);
            gpu_data.particles.push(ParticleGpu {
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

        // entities と particles の対応が一致していることを確認
        assert_eq!(gpu_data.entities.len(), gpu_data.particles.len());
        assert_eq!(gpu_data.entities.len(), 10);

        // 各エンティティが正しい粒子データと対応していること
        for (i, entity) in gpu_data.entities.iter().enumerate() {
            assert_eq!(*entity, entities[i]);
            assert_eq!(gpu_data.particles[i].pos[0], i as f32 * 0.1);
        }
    }

    #[test]
    fn test_readback_data_maps_to_correct_entities() {
        use crate::gpu::plugin::GpuParticleData;
        use crate::gpu::readback::GpuReadbackBuffer;
        use bevy::prelude::*;

        // GPU readback データが正しいエンティティにマッピングされることを検証
        let mut gpu_data = GpuParticleData::default();
        let readback = GpuReadbackBuffer::default();

        // 5つの粒子を作成
        let mut world = World::new();
        let entities: Vec<Entity> = (0..5).map(|_| world.spawn_empty().id()).collect();

        for (i, &entity) in entities.iter().enumerate() {
            gpu_data.entities.push(entity);
            gpu_data.particles.push(ParticleGpu {
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
        for (i, p) in gpu_data.particles.iter().enumerate() {
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
        for (i, entity) in gpu_data.entities.iter().enumerate() {
            assert_eq!(*entity, entities[i], "Entity mismatch at index {}", i);
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
        gpu_data.particles.clear();
        gpu_data.entities.clear();

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

        if dist_sq >= r_sum * r_sum || dist_sq < 1e-10 {
            return [0.0, 0.0, 0.0];
        }

        let dist = dist_sq.sqrt();
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
        let a = [force[0] * mass_inv, force[1] * mass_inv, force[2] * mass_inv];
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
            params.0, params.1, params.2, params.3,
        );

        // 法線方向の力が非負（引力にならない）ことを検証
        // n = (pos_p - pos_q) / dist = [-1, 0, 0]
        let delta = [pos_p[0] - pos_q[0], pos_p[1] - pos_q[1], pos_p[2] - pos_q[2]];
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
                pos_a, vel_a, radius, mass_inv,
                pos_b, vel_b, radius, mass_inv,
                params.0, params.1, params.2, params.3,
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
            initial_ke, max_ke, max_ke / initial_ke
        );

        // 最終エネルギーも初期以下（反発係数 0.5 で減衰）
        assert!(
            final_ke <= initial_ke * 1.01,
            "Final KE should not exceed initial: initial={:.6}, final={:.6}, ratio={:.4}",
            initial_ke, final_ke, final_ke / initial_ke
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
                pos_a, vel_a, radius, mass_inv,
                pos_b, vel_b, radius, mass_inv,
                params.0, params.1, params.2, params.3,
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
            initial_ke, max_ke, max_ke / initial_ke
        );

        assert!(
            final_ke <= initial_ke * 1.01,
            "Oblique final KE should not exceed initial: initial={:.6}, final={:.6}, ratio={:.4}",
            initial_ke, final_ke, final_ke / initial_ke
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

        let initial_ke =
            kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);
        let mut max_ke = initial_ke;

        for _ in 0..5000 {
            let force_on_a = gpu_compute_contact_force(
                pos_a, vel_a, radius_large, mass_inv_large,
                pos_b, vel_b, radius_small, mass_inv_small,
                params.0, params.1, params.2, params.3,
            );
            let force_on_b = [-force_on_a[0], -force_on_a[1], -force_on_a[2]];

            gpu_integrate_step(&mut pos_a, &mut vel_a, force_on_a, mass_inv_large, dt);
            gpu_integrate_step(&mut pos_b, &mut vel_b, force_on_b, mass_inv_small, dt);

            let ke =
                kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);
            if ke > max_ke {
                max_ke = ke;
            }
        }

        let final_ke =
            kinetic_energy(vel_a, mass_inv_large) + kinetic_energy(vel_b, mass_inv_small);

        assert!(
            max_ke <= initial_ke * 1.01,
            "Different-size max KE should not exceed initial: initial={:.6}, max={:.6}, ratio={:.4}",
            initial_ke, max_ke, max_ke / initial_ke
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
            params.0, params.1, params.2, params.3,
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
            params.0, params.1, params.2, params.3,
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
            [f32; 3], [f32; 3], f32, f32,
            [f32; 3], [f32; 3], f32, f32,
            &str,
        )> = vec![
            // Case 1: 同サイズ、静止、浅い接触
            (
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                [0.019, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                "same-size stationary shallow contact",
            ),
            // Case 2: 同サイズ、接近中
            (
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                [0.019, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                "same-size approaching",
            ),
            // Case 3: 同サイズ、離反中
            (
                [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                [0.019, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01, 1.0 / 95.493,
                "same-size separating",
            ),
            // Case 4: 異サイズ、接近中
            (
                [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 0.01, 1.0 / 95.493,
                [0.013, 0.0, 0.0], [-0.3, 0.0, 0.0], 0.004, 1.0 / 1492.077,
                "different-size approaching",
            ),
            // Case 5: 斜め接触、接線速度あり
            (
                [0.0, 0.0, 0.0], [1.0, 0.5, 0.0], 0.01, 1.0 / 95.493,
                [0.019, 0.001, 0.0], [-1.0, -0.5, 0.0], 0.01, 1.0 / 95.493,
                "oblique with tangential velocity",
            ),
        ];

        for (pos_p, vel_p, radius_p, mass_p, pos_q, vel_q, radius_q, mass_q, label) in &cases {
            let cpu_force = cpu_compute_contact_force(
                *pos_p, *vel_p, *radius_p, *mass_p,
                *pos_q, *vel_q, *radius_q, *mass_q,
                &material,
            );

            let gpu_force = gpu_compute_contact_force(
                *pos_p, *vel_p, *radius_p, 1.0 / mass_p,
                *pos_q, *vel_q, *radius_q, 1.0 / mass_q,
                gpu_params.0, gpu_params.1, gpu_params.2, gpu_params.3,
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
                    label, cos_sim
                );

                // 大きさの差異を報告（CPU の force clamp の影響がなければ一致するはず）
                assert!(
                    (mag_ratio - 1.0).abs() < 0.05,
                    "[{}] Force magnitude differs: CPU={:.4}, GPU={:.4}, ratio={:.4}",
                    label, cpu_mag, gpu_mag, mag_ratio
                );
            } else {
                // 両方ゼロなら OK
                assert!(
                    cpu_mag < 1e-6 && gpu_mag < 1e-6,
                    "[{}] One force is zero but not the other: CPU_mag={:.6}, GPU_mag={:.6}",
                    label, cpu_mag, gpu_mag
                );
            }
        }
    }

    /// CPU と GPU でシミュレーションを N ステップ走らせて軌跡を比較するテスト
    #[test]
    fn test_cpu_gpu_trajectory_consistency() {
        let dt = 1.0 / 5000.0; // SimulationTime::default().dt
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
                cpu_pos_a, cpu_vel_a, radius, mass,
                cpu_pos_b, cpu_vel_b, radius, mass,
                &material,
            );
            let cpu_force_b = [-cpu_force[0], -cpu_force[1], -cpu_force[2]];
            gpu_integrate_step(&mut cpu_pos_a, &mut cpu_vel_a, cpu_force, mass_inv, dt);
            gpu_integrate_step(&mut cpu_pos_b, &mut cpu_vel_b, cpu_force_b, mass_inv, dt);

            // GPU step
            let gpu_force = gpu_compute_contact_force(
                gpu_pos_a, gpu_vel_a, radius, mass_inv,
                gpu_pos_b, gpu_vel_b, radius, mass_inv,
                gpu_params.0, gpu_params.1, gpu_params.2, gpu_params.3,
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
            cpu_pos_a[0], cpu_pos_a[1], cpu_pos_a[2],
            cpu_vel_a[0], cpu_vel_a[1], cpu_vel_a[2],
        );
        println!(
            "Final: GPU pos_a=[{:.6}, {:.6}, {:.6}], vel_a=[{:.6}, {:.6}, {:.6}]",
            gpu_pos_a[0], gpu_pos_a[1], gpu_pos_a[2],
            gpu_vel_a[0], gpu_vel_a[1], gpu_vel_a[2],
        );
        println!(
            "Max pos diff: {:.10}, max vel diff: {:.10}",
            max_pos_diff, max_vel_diff
        );

        // 位置の差は粒子半径の 1% 以内であるべき
        assert!(
            max_pos_diff < radius * 0.01,
            "Position diverged too much: max_pos_diff={:.8} (threshold={:.8})",
            max_pos_diff, radius * 0.01
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
}
