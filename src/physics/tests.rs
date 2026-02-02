#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use crate::physics::collision::{compute_wall_contact_force, WallProperties};
    use crate::physics::contact::{compute_particle_contact_force, ContactState, MaterialProperties};
    use crate::physics::integrator::{
        clamp_to_container, clamp_velocity, integrate_first_half, integrate_second_half,
    };
    use crate::simulation::Container;

    /// 壁との衝突力テスト：床に接触した粒子は上向きの力を受ける
    #[test]
    fn test_wall_contact_floor() {
        let container = Container::default();
        let wall_props = WallProperties::default();

        // 床に少しめり込んだ位置
        let floor_y = container.base_position.y - container.half_extents.y;
        let radius = 0.02;
        let mass = 0.084; // 大粒子の質量
        let pos = Vec3::new(0.0, floor_y + radius * 0.9, 0.0); // 床に10%めり込み
        let vel = Vec3::new(0.0, -1.0, 0.0); // 下向きに移動中
        let omega = Vec3::ZERO;

        let result = compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

        // 上向きの力を受けるはず
        assert!(
            result.force.y > 0.0,
            "Floor contact should produce upward force, got: {:?}",
            result.force
        );

        // 力が有限値であること
        assert!(result.force.is_finite(), "Force should be finite");
        assert!(result.torque.is_finite(), "Torque should be finite");
    }

    /// 壁との衝突力テスト：壁に接触していない粒子は力を受けない
    #[test]
    fn test_wall_no_contact() {
        let container = Container::default();
        let wall_props = WallProperties::default();

        // コンテナの中央
        let radius = 0.02;
        let mass = 0.084;
        let pos = container.base_position;
        let vel = Vec3::ZERO;
        let omega = Vec3::ZERO;

        let result = compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

        assert!(
            result.force.length() < 1e-6,
            "No contact should produce zero force, got: {:?}",
            result.force
        );
    }

    /// 粒子間衝突力テスト：衝突した粒子は反発力を受ける
    #[test]
    fn test_particle_collision() {
        let material = MaterialProperties::default();
        let mut contact_state = ContactState::default();
        let dt = 1.0 / 120.0;

        let radius = 0.02;
        let mass = 0.084;

        // 2粒子が少し重なった状態
        let pos1 = Vec3::new(0.0, 0.0, 0.0);
        let pos2 = Vec3::new(0.035, 0.0, 0.0); // 0.035 < 0.04 (2*radius) なので重なり

        let vel1 = Vec3::new(1.0, 0.0, 0.0); // 右に移動
        let vel2 = Vec3::new(-1.0, 0.0, 0.0); // 左に移動（衝突方向）

        let (force1, force2) = compute_particle_contact_force(
            pos1,
            vel1,
            Vec3::ZERO,
            radius,
            mass,
            pos2,
            vel2,
            Vec3::ZERO,
            radius,
            mass,
            &material,
            &mut contact_state,
            dt,
        );

        // 粒子1は左向きの力を受ける（反発）
        assert!(
            force1.force.x < 0.0,
            "Particle 1 should receive leftward force, got: {:?}",
            force1.force
        );

        // 粒子2は右向きの力を受ける（反発）
        assert!(
            force2.force.x > 0.0,
            "Particle 2 should receive rightward force, got: {:?}",
            force2.force
        );

        // 作用反作用の法則
        let force_sum = force1.force + force2.force;
        assert!(
            force_sum.length() < 1e-6,
            "Forces should be equal and opposite, sum: {:?}",
            force_sum
        );

        // 力が有限値であること
        assert!(force1.force.is_finite(), "Force 1 should be finite");
        assert!(force2.force.is_finite(), "Force 2 should be finite");
    }

    /// 粒子間衝突力テスト：衝突していない粒子は力を受けない
    #[test]
    fn test_particle_no_collision() {
        let material = MaterialProperties::default();
        let mut contact_state = ContactState::default();
        let dt = 1.0 / 120.0;

        let radius = 0.02;
        let mass = 0.084;

        // 2粒子が離れた状態
        let pos1 = Vec3::new(0.0, 0.0, 0.0);
        let pos2 = Vec3::new(0.1, 0.0, 0.0); // 十分離れている

        let (force1, force2) = compute_particle_contact_force(
            pos1,
            Vec3::ZERO,
            Vec3::ZERO,
            radius,
            mass,
            pos2,
            Vec3::ZERO,
            Vec3::ZERO,
            radius,
            mass,
            &material,
            &mut contact_state,
            dt,
        );

        assert!(
            force1.force.length() < 1e-6,
            "No collision should produce zero force, got: {:?}",
            force1.force
        );
        assert!(
            force2.force.length() < 1e-6,
            "No collision should produce zero force, got: {:?}",
            force2.force
        );
    }

    /// 積分テスト：重力下での自由落下
    #[test]
    fn test_integration_free_fall() {
        let dt = 1.0 / 120.0;
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let mass = 0.084;
        let inertia = 0.001;

        let mut pos = Vec3::new(0.0, 1.0, 0.0);
        let mut vel = Vec3::ZERO;
        let mut omega = Vec3::ZERO;
        let force = Vec3::ZERO;
        let torque = Vec3::ZERO;

        let initial_pos = pos;

        // 1ステップ積分
        integrate_first_half(&mut pos, &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt);
        integrate_second_half(&mut vel, &mut omega, force, torque, mass, inertia, gravity, dt);

        // 下向きに速度が増加
        assert!(vel.y < 0.0, "Velocity should be downward, got: {:?}", vel);

        // 位置が下がる
        assert!(
            pos.y < initial_pos.y,
            "Position should decrease, got: {:?}",
            pos
        );

        // 値が有限であること
        assert!(pos.is_finite(), "Position should be finite");
        assert!(vel.is_finite(), "Velocity should be finite");
    }

    /// 積分テスト：力によって加速する
    #[test]
    fn test_integration_with_force() {
        let dt = 1.0 / 120.0;
        let gravity = Vec3::ZERO;
        let mass = 0.084;
        let inertia = 0.001;

        let mut pos = Vec3::ZERO;
        let mut vel = Vec3::ZERO;
        let mut omega = Vec3::ZERO;
        let force = Vec3::new(1.0, 0.0, 0.0); // X方向に1Nの力
        let torque = Vec3::ZERO;

        integrate_first_half(&mut pos, &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt);
        integrate_second_half(&mut vel, &mut omega, force, torque, mass, inertia, gravity, dt);

        // X方向に加速
        assert!(vel.x > 0.0, "Velocity should increase in X, got: {:?}", vel);

        // 加速度 = F/m = 1/0.084 ≈ 11.9 m/s²
        // 速度変化 ≈ 11.9 * (1/120) ≈ 0.099 m/s
        let expected_vel = 1.0 / mass * dt;
        assert!(
            (vel.x - expected_vel).abs() < 0.01,
            "Velocity should match expected, got: {}, expected: {}",
            vel.x,
            expected_vel
        );
    }

    /// 数値安定性テスト：大きなオーバーラップでも力が有限
    #[test]
    fn test_numerical_stability_large_overlap() {
        let container = Container::default();
        let wall_props = WallProperties::default();

        // 大きくめり込んだ位置
        let floor_y = container.base_position.y - container.half_extents.y;
        let radius = 0.02;
        let mass = 0.084;
        let pos = Vec3::new(0.0, floor_y - radius, 0.0); // 完全に床の下
        let vel = Vec3::new(0.0, -10.0, 0.0); // 高速で下向き
        let omega = Vec3::ZERO;

        let result = compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

        // 力が有限値であること
        assert!(
            result.force.is_finite(),
            "Force should be finite even with large overlap, got: {:?}",
            result.force
        );

        // 加速度が妥当な範囲（壁は5000 m/s²以下）
        let accel = result.force.length() / mass;
        assert!(
            accel <= 5000.0 + 1e-6,
            "Acceleration should be clamped, got: {}",
            accel
        );
    }

    /// 数値安定性テスト：粒子同士の大きなオーバーラップ
    #[test]
    fn test_numerical_stability_particle_overlap() {
        let material = MaterialProperties::default();
        let mut contact_state = ContactState::default();
        let dt = 1.0 / 120.0;

        let radius = 0.02;
        let mass = 0.084;

        // ほぼ同じ位置（大きなオーバーラップ）
        let pos1 = Vec3::new(0.0, 0.0, 0.0);
        let pos2 = Vec3::new(0.01, 0.0, 0.0); // 半径の半分程度

        let (force1, force2) = compute_particle_contact_force(
            pos1,
            Vec3::ZERO,
            Vec3::ZERO,
            radius,
            mass,
            pos2,
            Vec3::ZERO,
            Vec3::ZERO,
            radius,
            mass,
            &material,
            &mut contact_state,
            dt,
        );

        // 力が有限値であること
        assert!(
            force1.force.is_finite(),
            "Force 1 should be finite, got: {:?}",
            force1.force
        );
        assert!(
            force2.force.is_finite(),
            "Force 2 should be finite, got: {:?}",
            force2.force
        );

        // 加速度が妥当な範囲
        let accel1 = force1.force.length() / mass;
        assert!(
            accel1 <= 1000.0 + 1e-6,
            "Acceleration 1 should be clamped, got: {}",
            accel1
        );
    }

    /// 粒子がコンテナ内に収まることをシミュレーションで確認
    #[test]
    fn test_particle_stays_in_container() {
        let container = Container::default();
        let wall_props = WallProperties::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 120.0;

        let radius = 0.02;
        let mass = 0.084;
        let inertia = (2.0 / 5.0) * mass * radius * radius;

        // コンテナ内の上部から落下開始
        let mut pos = Vec3::new(0.0, container.base_position.y + 0.1, 0.0);
        let mut vel = Vec3::ZERO;
        let mut omega = Vec3::ZERO;

        // コンテナの境界を計算
        let box_min = container.base_position - container.half_extents;
        let box_max = container.base_position + container.half_extents;

        // 5秒間（600ステップ）シミュレーション - 粒子が落ち着くまで
        // 実際のシミュレーションと同じ順序：力計算→位置更新→力計算→速度更新
        for step in 0..600 {
            // 前半：力計算→位置更新
            let wall_force1 = compute_wall_contact_force(
                pos, vel, omega, radius, mass, &container, &wall_props
            );
            integrate_first_half(
                &mut pos, &mut vel, &mut omega,
                wall_force1.force, wall_force1.torque,
                mass, inertia, gravity, dt
            );
            // 位置と速度をクランプ
            clamp_to_container(&mut pos, &mut vel, radius, box_min, box_max);
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

            // 後半：力計算→速度更新
            let wall_force2 = compute_wall_contact_force(
                pos, vel, omega, radius, mass, &container, &wall_props
            );
            integrate_second_half(
                &mut vel, &mut omega,
                wall_force2.force, wall_force2.torque,
                mass, inertia, gravity, dt
            );
            // 速度をクランプ
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);


            // 値が有限であることを確認
            assert!(
                pos.is_finite(),
                "Position became NaN/Inf at step {}: {:?}",
                step, pos
            );
            assert!(
                vel.is_finite(),
                "Velocity became NaN/Inf at step {}: {:?}",
                step, vel
            );

            // 粒子がコンテナ内に収まっていることを確認（少しのマージンを許容）
            let margin = radius * 2.0; // 粒子半径の2倍のマージン
            assert!(
                pos.x >= box_min.x - margin && pos.x <= box_max.x + margin,
                "Particle escaped X bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step, pos, box_min.x, box_max.x
            );
            assert!(
                pos.y >= box_min.y - margin && pos.y <= box_max.y + margin,
                "Particle escaped Y bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step, pos, box_min.y, box_max.y
            );
            assert!(
                pos.z >= box_min.z - margin && pos.z <= box_max.z + margin,
                "Particle escaped Z bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step, pos, box_min.z, box_max.z
            );
        }

        // 最終的にコンテナ内に収まっていることを確認（跳ね返り中の場合もある）
        assert!(
            pos.y >= box_min.y - radius && pos.y <= box_max.y + radius,
            "Particle should be within container at end, got y={}, bounds=[{}, {}]",
            pos.y, box_min.y, box_max.y
        );

        // 速度が発散していないことを確認
        assert!(
            vel.length() < 10.0,
            "Velocity should be reasonable, got {:?}",
            vel
        );
    }

    /// 複数粒子の初期配置から時間発展しても箱から飛び出さないことをテスト
    #[test]
    fn test_multiple_particles_stay_in_container() {
        use crate::rendering::SimulationConfig;
        use rand::Rng;
        use std::f32::consts::PI;

        let container = Container::default();
        let wall_props = WallProperties::default();
        let material = MaterialProperties::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 120.0;
        let config = SimulationConfig::default();

        // 粒子の状態を保持する構造体
        struct Particle {
            pos: Vec3,
            vel: Vec3,
            omega: Vec3,
            radius: f32,
            mass: f32,
            inertia: f32,
        }

        let mut rng = rand::rng();
        let mut particles: Vec<Particle> = Vec::new();

        // 実際のスポーンコードと同じロジックで粒子を生成
        let spawn_area_x = container.half_extents.x - config.large_radius;
        let spawn_area_z = container.half_extents.z - config.large_radius;
        let base_y = container.base_position.y - container.half_extents.y;

        // 大粒子
        for _ in 0..config.num_large {
            let x = rng.random_range(-spawn_area_x..spawn_area_x);
            let z = rng.random_range(-spawn_area_z..spawn_area_z);
            let y = base_y + config.large_radius + rng.random_range(0.0..0.2);

            let volume = (4.0 / 3.0) * PI * config.large_radius.powi(3);
            let mass = config.density * volume;
            let inertia = (2.0 / 5.0) * mass * config.large_radius.powi(2);

            particles.push(Particle {
                pos: Vec3::new(x, y, z),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
                radius: config.large_radius,
                mass,
                inertia,
            });
        }

        // 小粒子
        for _ in 0..config.num_small {
            let x = rng.random_range(-spawn_area_x..spawn_area_x);
            let z = rng.random_range(-spawn_area_z..spawn_area_z);
            let y = base_y + config.small_radius + rng.random_range(0.0..0.2);

            let volume = (4.0 / 3.0) * PI * config.small_radius.powi(3);
            let mass = config.density * volume;
            let inertia = (2.0 / 5.0) * mass * config.small_radius.powi(2);

            particles.push(Particle {
                pos: Vec3::new(x, y, z),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
                radius: config.small_radius,
                mass,
                inertia,
            });
        }

        let box_min = container.base_position - container.half_extents;
        let box_max = container.base_position + container.half_extents;

        println!("Testing {} particles in container [{:?} to {:?}]",
            particles.len(), box_min, box_max);

        // 接触状態を管理
        let mut contact_states: std::collections::HashMap<(usize, usize), ContactState> =
            std::collections::HashMap::new();

        // 1秒間（120ステップ）シミュレーション
        for step in 0..120 {
            // 各粒子に対して力を計算して積分
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            // 壁との衝突力
            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos, p.vel, p.omega, p.radius, p.mass, &container, &wall_props
                );
                forces[i] += wall_force.force;
                torques[i] += wall_force.torque;
            }

            // 粒子間の衝突力（全ペアをチェック - テスト用に簡略化）
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];

                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos, pi.vel, pi.omega, pi.radius, pi.mass,
                        pj.pos, pj.vel, pj.omega, pj.radius, pj.mass,
                        &material, contact_state, dt
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // 積分（前半）
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_first_half(
                    &mut p.pos, &mut p.vel, &mut p.omega,
                    forces[i], torques[i], p.mass, p.inertia, gravity, dt
                );
                // 位置と速度をクランプ
                clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            // 力を再計算
            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos, p.vel, p.omega, p.radius, p.mass, &container, &wall_props
                );
                forces[i] += wall_force.force;
                torques[i] += wall_force.torque;
            }

            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];

                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos, pi.vel, pi.omega, pi.radius, pi.mass,
                        pj.pos, pj.vel, pj.omega, pj.radius, pj.mass,
                        &material, contact_state, dt
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // 積分（後半）
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_second_half(
                    &mut p.vel, &mut p.omega,
                    forces[i], torques[i], p.mass, p.inertia, gravity, dt
                );
                // 速度をクランプ
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            // 全粒子が箱内に収まっていることを確認
            for (i, p) in particles.iter().enumerate() {
                let margin = p.radius * 3.0; // マージンを少し広げる

                assert!(
                    p.pos.is_finite(),
                    "Particle {} position became NaN/Inf at step {}: {:?}",
                    i, step, p.pos
                );
                assert!(
                    p.vel.is_finite(),
                    "Particle {} velocity became NaN/Inf at step {}: {:?}",
                    i, step, p.vel
                );

                if p.pos.x < box_min.x - margin || p.pos.x > box_max.x + margin {
                    panic!(
                        "Particle {} escaped X bounds at step {}: pos={:?}, vel={:?}, bounds=[{}, {}]",
                        i, step, p.pos, p.vel, box_min.x, box_max.x
                    );
                }
                if p.pos.y < box_min.y - margin || p.pos.y > box_max.y + margin {
                    panic!(
                        "Particle {} escaped Y bounds at step {}: pos={:?}, vel={:?}, bounds=[{}, {}]",
                        i, step, p.pos, p.vel, box_min.y, box_max.y
                    );
                }
                if p.pos.z < box_min.z - margin || p.pos.z > box_max.z + margin {
                    panic!(
                        "Particle {} escaped Z bounds at step {}: pos={:?}, vel={:?}, bounds=[{}, {}]",
                        i, step, p.pos, p.vel, box_min.z, box_max.z
                    );
                }
            }

            // 進捗表示
            if step % 30 == 0 {
                let max_vel = particles.iter().map(|p| p.vel.length()).fold(0.0f32, f32::max);
                let min_y = particles.iter().map(|p| p.pos.y).fold(f32::MAX, f32::min);
                let max_y = particles.iter().map(|p| p.pos.y).fold(f32::MIN, f32::max);
                println!("Step {}: Y range [{:.3}, {:.3}], max vel {:.2}",
                    step, min_y, max_y, max_vel);
            }
        }

        println!("All {} particles stayed within container bounds!", particles.len());
    }

    /// コンテナサイズと粒子数の関係をテスト
    #[test]
    fn test_container_capacity() {
        use crate::rendering::SimulationConfig;
        use std::f32::consts::PI;

        let container = Container::default();
        let config = SimulationConfig::default();

        // コンテナの体積
        let container_volume = container.half_extents.x * 2.0
            * container.half_extents.y * 2.0
            * container.half_extents.z * 2.0;

        // 粒子の総体積
        let large_volume = (4.0 / 3.0) * PI * config.large_radius.powi(3);
        let small_volume = (4.0 / 3.0) * PI * config.small_radius.powi(3);
        let total_particle_volume =
            config.num_large as f32 * large_volume + config.num_small as f32 * small_volume;

        // 充填率（ランダム充填の理論最大は約64%）
        let packing_fraction = total_particle_volume / container_volume;

        println!("Container volume: {} m³ ({} liters)", container_volume, container_volume * 1000.0);
        println!("Total particle volume: {} m³", total_particle_volume);
        println!("Packing fraction: {:.1}%", packing_fraction * 100.0);

        // 充填率が40%以下であることを確認（余裕を持たせる）
        assert!(
            packing_fraction < 0.4,
            "Packing fraction too high: {:.1}% (should be < 40%)",
            packing_fraction * 100.0
        );
    }
}
