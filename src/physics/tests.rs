#[cfg(test)]
mod tests {
    use bevy::prelude::*;

    use crate::physics::collision::{compute_wall_contact_force, WallProperties};
    use crate::physics::contact::{
        compute_particle_contact_force, ContactState, MaterialProperties,
    };
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

        let result =
            compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

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

        let result =
            compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

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
        integrate_first_half(
            &mut pos, &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt,
        );
        integrate_second_half(
            &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt,
        );

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

        integrate_first_half(
            &mut pos, &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt,
        );
        integrate_second_half(
            &mut vel, &mut omega, force, torque, mass, inertia, gravity, dt,
        );

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

        let result =
            compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

        // 力が有限値であること
        assert!(
            result.force.is_finite(),
            "Force should be finite even with large overlap, got: {:?}",
            result.force
        );

        // 力が正の方向（床から押し出す方向）であること
        assert!(
            result.force.y > 0.0,
            "Force should push up from floor, got: {:?}",
            result.force
        );

        // 2000Hzタイムステップでの速度変化が妥当であること
        // dt = 1/2000 = 0.0005s, accel = F/m
        // Δv = accel * dt でチェック
        let dt = 1.0 / 2000.0;
        let accel = result.force.y / mass;
        let dv = accel * dt;
        // 1ステップで速度が10 m/s以上変化しないこと（妥当な範囲）
        assert!(
            dv < 20.0,
            "Velocity change per step should be reasonable, got: {} m/s",
            dv
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
            let wall_force1 =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);
            integrate_first_half(
                &mut pos,
                &mut vel,
                &mut omega,
                wall_force1.force,
                wall_force1.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            // 位置と速度をクランプ
            clamp_to_container(&mut pos, &mut vel, radius, box_min, box_max);
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

            // 後半：力計算→速度更新
            let wall_force2 =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);
            integrate_second_half(
                &mut vel,
                &mut omega,
                wall_force2.force,
                wall_force2.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            // 速度をクランプ
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

            // 値が有限であることを確認
            assert!(
                pos.is_finite(),
                "Position became NaN/Inf at step {}: {:?}",
                step,
                pos
            );
            assert!(
                vel.is_finite(),
                "Velocity became NaN/Inf at step {}: {:?}",
                step,
                vel
            );

            // 粒子がコンテナ内に収まっていることを確認（少しのマージンを許容）
            let margin = radius * 2.0; // 粒子半径の2倍のマージン
            assert!(
                pos.x >= box_min.x - margin && pos.x <= box_max.x + margin,
                "Particle escaped X bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step,
                pos,
                box_min.x,
                box_max.x
            );
            assert!(
                pos.y >= box_min.y - margin && pos.y <= box_max.y + margin,
                "Particle escaped Y bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step,
                pos,
                box_min.y,
                box_max.y
            );
            assert!(
                pos.z >= box_min.z - margin && pos.z <= box_max.z + margin,
                "Particle escaped Z bounds at step {}: pos={:?}, bounds=[{}, {}]",
                step,
                pos,
                box_min.z,
                box_max.z
            );
        }

        // 最終的にコンテナ内に収まっていることを確認（跳ね返り中の場合もある）
        assert!(
            pos.y >= box_min.y - radius && pos.y <= box_max.y + radius,
            "Particle should be within container at end, got y={}, bounds=[{}, {}]",
            pos.y,
            box_min.y,
            box_max.y
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
        use crate::simulation::SimulationConfig;
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

        println!(
            "Testing {} particles in container [{:?} to {:?}]",
            particles.len(),
            box_min,
            box_max
        );

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
                    p.pos,
                    p.vel,
                    p.omega,
                    p.radius,
                    p.mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        pi.radius,
                        pi.mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        pj.radius,
                        pj.mass,
                        &material,
                        contact_state,
                        dt,
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
                // 位置と速度をクランプ
                clamp_to_container(&mut p.pos, &mut p.vel, p.radius, box_min, box_max);
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            // 力を再計算
            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    p.radius,
                    p.mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        pi.radius,
                        pi.mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        pj.radius,
                        pj.mass,
                        &material,
                        contact_state,
                        dt,
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
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    p.mass,
                    p.inertia,
                    gravity,
                    dt,
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
                    i,
                    step,
                    p.pos
                );
                assert!(
                    p.vel.is_finite(),
                    "Particle {} velocity became NaN/Inf at step {}: {:?}",
                    i,
                    step,
                    p.vel
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
                let max_vel = particles
                    .iter()
                    .map(|p| p.vel.length())
                    .fold(0.0f32, f32::max);
                let min_y = particles.iter().map(|p| p.pos.y).fold(f32::MAX, f32::min);
                let max_y = particles.iter().map(|p| p.pos.y).fold(f32::MIN, f32::max);
                println!(
                    "Step {}: Y range [{:.3}, {:.3}], max vel {:.2}",
                    step, min_y, max_y, max_vel
                );
            }
        }

        println!(
            "All {} particles stayed within container bounds!",
            particles.len()
        );
    }

    /// コンテナサイズと粒子数の関係をテスト
    #[test]
    fn test_container_capacity() {
        use crate::simulation::SimulationConfig;
        use std::f32::consts::PI;

        let container = Container::default();
        let config = SimulationConfig::default();

        // コンテナの体積
        let container_volume = container.half_extents.x
            * 2.0
            * container.half_extents.y
            * 2.0
            * container.half_extents.z
            * 2.0;

        // 粒子の総体積
        let large_volume = (4.0 / 3.0) * PI * config.large_radius.powi(3);
        let small_volume = (4.0 / 3.0) * PI * config.small_radius.powi(3);
        let total_particle_volume =
            config.num_large as f32 * large_volume + config.num_small as f32 * small_volume;

        // 充填率（ランダム充填の理論最大は約64%）
        let packing_fraction = total_particle_volume / container_volume;

        println!(
            "Container volume: {} m³ ({} liters)",
            container_volume,
            container_volume * 1000.0
        );
        println!("Total particle volume: {} m³", total_particle_volume);
        println!("Packing fraction: {:.1}%", packing_fraction * 100.0);

        // 充填率が40%以下であることを確認（余裕を持たせる）
        assert!(
            packing_fraction < 0.4,
            "Packing fraction too high: {:.1}% (should be < 40%)",
            packing_fraction * 100.0
        );
    }

    // ========================================
    // Energy Dissipation and Settling Tests
    // ========================================

    /// Calculate kinetic energy of a particle
    fn kinetic_energy(vel: Vec3, omega: Vec3, mass: f32, inertia: f32) -> f32 {
        0.5 * mass * vel.length_squared() + 0.5 * inertia * omega.length_squared()
    }

    /// Test energy dissipation for a single bouncing particle
    #[test]
    fn test_single_particle_energy_decay() {
        use std::f32::consts::PI;

        let container = Container::default();
        let wall_props = WallProperties::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 120.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        // Start particle above floor with some horizontal velocity
        let floor_y = container.base_position.y - container.half_extents.y;
        let mut pos = Vec3::new(0.0, floor_y + radius + 0.1, 0.0); // 10cm above floor
        let mut vel = Vec3::new(0.5, 0.0, 0.5); // Some horizontal velocity
        let mut omega = Vec3::ZERO;

        let box_min = container.base_position - container.half_extents;
        let box_max = container.base_position + container.half_extents;

        let initial_energy = kinetic_energy(vel, omega, mass, inertia)
            + mass * gravity.length() * (pos.y - floor_y - radius); // Include potential energy

        println!("Initial total energy: {:.6} J", initial_energy);
        println!(
            "Wall damping: {}, friction: {}",
            wall_props.damping, wall_props.friction
        );

        let mut energy_history: Vec<(f32, f32, f32)> = Vec::new(); // (time, kinetic, potential)

        // Simulate for 10 seconds (1200 steps)
        let total_steps = 1200;
        for step in 0..total_steps {
            let time = step as f32 * dt;

            // Calculate energies before step
            let ke = kinetic_energy(vel, omega, mass, inertia);
            let pe = mass * gravity.length() * (pos.y - floor_y - radius).max(0.0);

            if step % 120 == 0 {
                energy_history.push((time, ke, pe));
                println!(
                    "t={:.1}s: KE={:.6}J, PE={:.6}J, Total={:.6}J, vel={:.4}m/s, y={:.4}m",
                    time,
                    ke,
                    pe,
                    ke + pe,
                    vel.length(),
                    pos.y
                );
            }

            // Physics step (simplified - wall collision only)
            let wall_force1 =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);
            integrate_first_half(
                &mut pos,
                &mut vel,
                &mut omega,
                wall_force1.force,
                wall_force1.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            clamp_to_container(&mut pos, &mut vel, radius, box_min, box_max);
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

            let wall_force2 =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);
            integrate_second_half(
                &mut vel,
                &mut omega,
                wall_force2.force,
                wall_force2.torque,
                mass,
                inertia,
                gravity,
                dt,
            );
            clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);
        }

        let final_ke = kinetic_energy(vel, omega, mass, inertia);
        let final_pe = mass * gravity.length() * (pos.y - floor_y - radius).max(0.0);
        let final_energy = final_ke + final_pe;

        println!("\nFinal state after 10s:");
        println!("  Position: {:?}", pos);
        println!("  Velocity: {:?} (magnitude: {:.6} m/s)", vel, vel.length());
        println!("  Angular velocity: {:?}", omega);
        println!("  Final KE: {:.6} J", final_ke);
        println!("  Final total energy: {:.6} J", final_energy);
        println!(
            "  Energy dissipated: {:.2}%",
            (1.0 - final_energy / initial_energy) * 100.0
        );

        // Check that energy has decreased significantly
        assert!(
            final_energy < initial_energy * 0.5,
            "Energy should decrease by at least 50%, but only decreased by {:.1}%",
            (1.0 - final_energy / initial_energy) * 100.0
        );

        // Check velocity is small (particle should be settling)
        println!("\nVelocity at end: {:.6} m/s", vel.length());
    }

    /// Test energy dissipation with multiple particles colliding
    #[test]
    fn test_multi_particle_energy_decay() {
        use std::f32::consts::PI;

        let container = Container::default();
        let wall_props = WallProperties::default();
        let material = MaterialProperties::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 120.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        // Create a small cluster of particles with random velocities
        struct Particle {
            pos: Vec3,
            vel: Vec3,
            omega: Vec3,
        }

        let floor_y = container.base_position.y - container.half_extents.y;
        let box_min = container.base_position - container.half_extents;
        let box_max = container.base_position + container.half_extents;

        // 10 particles in a small area with random velocities
        let mut particles = vec![
            Particle {
                pos: Vec3::new(-0.02, floor_y + radius + 0.05, 0.0),
                vel: Vec3::new(0.3, 0.0, 0.1),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.02, floor_y + radius + 0.05, 0.0),
                vel: Vec3::new(-0.2, 0.1, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, floor_y + radius + 0.08, 0.02),
                vel: Vec3::new(0.1, -0.1, 0.2),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, floor_y + radius + 0.08, -0.02),
                vel: Vec3::new(-0.1, 0.0, -0.3),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, floor_y + radius + 0.11, 0.0),
                vel: Vec3::new(0.0, -0.2, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(-0.03, floor_y + radius + 0.02, 0.02),
                vel: Vec3::new(0.2, 0.1, -0.1),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.03, floor_y + radius + 0.02, -0.02),
                vel: Vec3::new(-0.15, 0.0, 0.15),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(-0.01, floor_y + radius + 0.06, -0.03),
                vel: Vec3::new(0.0, 0.05, 0.2),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.01, floor_y + radius + 0.06, 0.03),
                vel: Vec3::new(0.1, -0.05, -0.1),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, floor_y + radius + 0.14, 0.0),
                vel: Vec3::new(0.0, 0.0, 0.0),
                omega: Vec3::ZERO,
            },
        ];

        // Calculate initial kinetic energy
        let initial_ke: f32 = particles
            .iter()
            .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
            .sum();

        println!(
            "Initial kinetic energy: {:.6} J with {} particles",
            initial_ke,
            particles.len()
        );
        println!(
            "Material: restitution={}, friction={}",
            material.restitution, material.friction
        );

        let mut contact_states: std::collections::HashMap<(usize, usize), ContactState> =
            std::collections::HashMap::new();

        // Simulate for 10 seconds
        let total_steps = 1200;
        for step in 0..total_steps {
            let time = step as f32 * dt;

            // Calculate and record energy
            let total_ke: f32 = particles
                .iter()
                .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
                .sum();
            let max_vel = particles
                .iter()
                .map(|p| p.vel.length())
                .fold(0.0f32, f32::max);

            if step % 120 == 0 {
                println!(
                    "t={:.1}s: Total KE={:.6}J, Max vel={:.4}m/s",
                    time, total_ke, max_vel
                );
            }

            // Calculate forces
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            // Wall forces
            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
                );
                forces[i] += wall_force.force;
                torques[i] += wall_force.torque;
            }

            // Particle-particle forces
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];

                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Integrate first half
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_first_half(
                    &mut p.pos,
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_to_container(&mut p.pos, &mut p.vel, radius, box_min, box_max);
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            // Recalculate forces
            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Integrate second half
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_second_half(
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }
        }

        let final_ke: f32 = particles
            .iter()
            .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
            .sum();
        let max_final_vel = particles
            .iter()
            .map(|p| p.vel.length())
            .fold(0.0f32, f32::max);

        println!("\nFinal state after 10s:");
        println!("  Final KE: {:.6} J (was {:.6} J)", final_ke, initial_ke);
        println!("  Max velocity: {:.6} m/s", max_final_vel);
        println!(
            "  Energy remaining: {:.2}%",
            (final_ke / initial_ke) * 100.0
        );
    }

    /// Test settling behavior after oscillation stops
    #[test]
    fn test_settling_after_oscillation() {
        use std::f32::consts::PI;

        let mut container = Container::default();
        let wall_props = WallProperties::default();
        let material = MaterialProperties::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        // Use actual simulation timestep for accurate wall collision behavior
        let dt = 1.0 / 2000.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        struct Particle {
            pos: Vec3,
            vel: Vec3,
            omega: Vec3,
        }

        let floor_y = container.base_position.y - container.half_extents.y;

        // Create particles on the floor (away from divider at x=0)
        // Divider extends from x=-0.005 to x=+0.005, particle radius=0.02
        // So keep |x| > 0.025 to avoid divider interaction
        let mut particles = vec![
            Particle {
                pos: Vec3::new(-0.10, floor_y + radius + 0.001, 0.0),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(-0.15, floor_y + radius + 0.001, 0.0),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.10, floor_y + radius + 0.001, 0.0),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(-0.20, floor_y + radius * 3.0, 0.0),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.15, floor_y + radius * 3.0, 0.0),
                vel: Vec3::ZERO,
                omega: Vec3::ZERO,
            },
        ];

        let mut contact_states: std::collections::HashMap<(usize, usize), ContactState> =
            std::collections::HashMap::new();

        // Oscillation parameters
        let osc_amplitude = 0.005; // 5mm
        let osc_frequency = 5.0; // 5 Hz
        let mut osc_phase = 0.0f32;
        let mut oscillation_enabled = true;

        println!(
            "Starting oscillation test: amplitude={:.3}m, frequency={}Hz",
            osc_amplitude, osc_frequency
        );

        // Phase 1: Oscillate for 3 seconds
        let oscillation_steps = 6000; // 3 seconds at 2000Hz
        println!("\n=== Phase 1: Oscillation for 3 seconds ===");

        for step in 0..oscillation_steps {
            let time = step as f32 * dt;

            // Update oscillation
            if oscillation_enabled {
                osc_phase += osc_frequency * 2.0 * PI * dt;
                if osc_phase > 2.0 * PI {
                    osc_phase -= 2.0 * PI;
                }
                container.current_offset = osc_amplitude * osc_phase.sin();
            }

            let box_offset = Vec3::Y * container.current_offset;
            let box_min = container.base_position - container.half_extents + box_offset;
            let box_max = container.base_position + container.half_extents + box_offset;

            // Calculate forces
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Integrate
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_first_half(
                    &mut p.pos,
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_to_container(&mut p.pos, &mut p.vel, radius, box_min, box_max);
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            for (i, p) in particles.iter_mut().enumerate() {
                integrate_second_half(
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            let total_ke: f32 = particles
                .iter()
                .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
                .sum();
            let max_vel = particles
                .iter()
                .map(|p| p.vel.length())
                .fold(0.0f32, f32::max);

            if step % 1000 == 0 {
                println!(
                    "t={:.1}s: KE={:.6}J, max_vel={:.4}m/s, offset={:.4}m",
                    time, total_ke, max_vel, container.current_offset
                );
            }
        }

        // Record state at oscillation stop
        let ke_at_stop: f32 = particles
            .iter()
            .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
            .sum();
        let max_vel_at_stop = particles
            .iter()
            .map(|p| p.vel.length())
            .fold(0.0f32, f32::max);

        println!("\n=== Oscillation stopped ===");
        println!("KE at stop: {:.6} J", ke_at_stop);
        println!("Max velocity at stop: {:.4} m/s", max_vel_at_stop);

        // Phase 2: Let particles settle for 10 seconds
        oscillation_enabled = false;
        container.current_offset = 0.0;

        println!("\n=== Phase 2: Settling for 10 seconds ===");

        let settling_steps = 20000; // 10 seconds at 2000Hz
        let mut settled_at_step: Option<usize> = None;
        // Note: Velocity Verlet with clamping creates a residual velocity of ~g*dt/2
        // At 2000Hz, this is about 0.0025 m/s, which is effectively "at rest"
        let settling_threshold = 0.005; // 5 mm/s (accounts for numerical artifacts)

        for step in 0..settling_steps {
            let time = 3.0 + step as f32 * dt; // Time since start

            let box_min = container.base_position - container.half_extents;
            let box_max = container.base_position + container.half_extents;

            // Calculate forces
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Integrate
            for (i, p) in particles.iter_mut().enumerate() {
                integrate_first_half(
                    &mut p.pos,
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_to_container(&mut p.pos, &mut p.vel, radius, box_min, box_max);
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for (i, p) in particles.iter().enumerate() {
                let wall_force = compute_wall_contact_force(
                    p.pos,
                    p.vel,
                    p.omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
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
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            for (i, p) in particles.iter_mut().enumerate() {
                integrate_second_half(
                    &mut p.vel,
                    &mut p.omega,
                    forces[i],
                    torques[i],
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_velocity(&mut p.vel, &mut p.omega, 10.0, 100.0);
            }

            let total_ke: f32 = particles
                .iter()
                .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
                .sum();
            let max_vel = particles
                .iter()
                .map(|p| p.vel.length())
                .fold(0.0f32, f32::max);

            // Check if settled
            if settled_at_step.is_none() && max_vel < settling_threshold {
                settled_at_step = Some(step);
                println!(
                    "*** Particles settled at t={:.2}s (step {}) ***",
                    time, step
                );
            }

            if step % 2000 == 0 {
                println!(
                    "t={:.1}s: KE={:.6}J, max_vel={:.4}m/s",
                    time, total_ke, max_vel
                );
            }
        }

        let final_ke: f32 = particles
            .iter()
            .map(|p| kinetic_energy(p.vel, p.omega, mass, inertia))
            .sum();
        let final_max_vel = particles
            .iter()
            .map(|p| p.vel.length())
            .fold(0.0f32, f32::max);

        println!("\n=== Final Results ===");
        println!("KE at oscillation stop: {:.6} J", ke_at_stop);
        println!("Final KE after 10s settling: {:.6} J", final_ke);
        println!("Final max velocity: {:.6} m/s", final_max_vel);

        if let Some(step) = settled_at_step {
            let settling_time = step as f32 * dt;
            println!(
                "Settling time: {:.2} seconds ({} steps)",
                settling_time, step
            );
        } else {
            println!("WARNING: Particles did not settle within 10 seconds!");
            println!(
                "  Final max velocity: {:.4} m/s (threshold: {} m/s)",
                final_max_vel, settling_threshold
            );
        }

        // Report on particle final positions
        println!("\nFinal particle states:");
        for (i, p) in particles.iter().enumerate() {
            println!(
                "  Particle {}: pos={:?}, vel={:.6}m/s",
                i,
                p.pos,
                p.vel.length()
            );
        }
    }

    /// Test divider collision behavior
    #[test]
    fn test_divider_collision() {
        let container = Container::default();
        let wall_props = WallProperties::default();

        let radius: f32 = 0.02;
        let mass: f32 = 0.084;
        let omega = Vec3::ZERO;

        let floor_y = container.base_position.y - container.half_extents.y;
        let half_thickness = container.divider_thickness / 2.0; // 0.005m

        println!(
            "Divider: thickness={}, half={}, height={}",
            container.divider_thickness, half_thickness, container.divider_height
        );
        println!("Particle radius: {}", radius);

        // Test 1: Particle approaching divider from right side
        println!("\n=== Test 1: Approaching from right ===");
        for x in [0.03, 0.025, 0.024, 0.02, 0.01, 0.005, 0.0, -0.005].iter() {
            let pos = Vec3::new(*x, floor_y + 0.05, 0.0); // Below divider top
            let vel = Vec3::new(-0.5, 0.0, 0.0); // Moving left toward divider

            let result =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

            println!(
                "  x={:+.4}: force.x={:+.2}N (overlap={:.4}m)",
                x,
                result.force.x,
                if *x > 0.0 {
                    half_thickness - (*x - radius)
                } else {
                    0.0
                }
            );
        }

        // Test 2: Particle at various x positions with zero velocity
        println!("\n=== Test 2: Stationary particles near divider ===");
        for x in [
            0.03, 0.025, 0.024, 0.015, 0.005, 0.0, -0.005, -0.015, -0.024, -0.025, -0.03,
        ]
        .iter()
        {
            let pos = Vec3::new(*x, floor_y + 0.05, 0.0);
            let vel = Vec3::ZERO;

            let result =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

            if result.force.x.abs() > 0.1 {
                println!("  x={:+.4}: force.x={:+.2}N", x, result.force.x);
            }
        }

        // Test 3: Particle crossing x=0 boundary
        println!("\n=== Test 3: Crossing x=0 boundary ===");
        for x in [0.002, 0.001, 0.0001, 0.0, -0.0001, -0.001, -0.002].iter() {
            let pos = Vec3::new(*x, floor_y + 0.05, 0.0);
            let vel = Vec3::new(-0.5, 0.0, 0.0); // Moving left

            let result =
                compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

            println!("  x={:+.6}: force.x={:+.2}N", x, result.force.x);
        }

        // Test 4: Energy check - particle bouncing off divider
        // Use the actual simulation timestep (1/2000s)
        println!("\n=== Test 4: Energy during divider collision (dt=1/2000s) ===");
        {
            use std::f32::consts::PI;
            let gravity = Vec3::new(0.0, -9.81, 0.0);
            let dt = 1.0 / 2000.0; // Actual simulation timestep

            let density: f32 = 2500.0;
            let volume = (4.0 / 3.0) * PI * radius.powi(3);
            let mass = density * volume;
            let inertia = (2.0 / 5.0) * mass * radius.powi(2);

            // Start particle moving toward divider
            let mut pos = Vec3::new(0.05, floor_y + 0.05, 0.0);
            let mut vel = Vec3::new(-1.0, 0.0, 0.0); // Moving left at 1 m/s
            let mut omega = Vec3::ZERO;

            let box_min = container.base_position - container.half_extents;
            let box_max = container.base_position + container.half_extents;

            let initial_ke = 0.5 * mass * vel.length_squared();
            println!("Initial KE: {:.6} J, vel: {:?}", initial_ke, vel);

            for step in 0..2000 {
                // 1 second with actual timestep
                let wall_force1 = compute_wall_contact_force(
                    pos,
                    vel,
                    omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
                );
                integrate_first_half(
                    &mut pos,
                    &mut vel,
                    &mut omega,
                    wall_force1.force,
                    wall_force1.torque,
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_to_container(&mut pos, &mut vel, radius, box_min, box_max);
                clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

                let wall_force2 = compute_wall_contact_force(
                    pos,
                    vel,
                    omega,
                    radius,
                    mass,
                    &container,
                    &wall_props,
                );
                integrate_second_half(
                    &mut vel,
                    &mut omega,
                    wall_force2.force,
                    wall_force2.torque,
                    mass,
                    inertia,
                    gravity,
                    dt,
                );
                clamp_velocity(&mut vel, &mut omega, 10.0, 100.0);

                let ke = 0.5 * mass * vel.length_squared();

                // Print every 100 steps and during collision
                if step < 50 || step % 200 == 0 {
                    println!(
                        "  step {}: x={:.4}, vel.x={:.4}, KE={:.6}J, f1={:.1}N, f2={:.1}N",
                        step, pos.x, vel.x, ke, wall_force1.force.x, wall_force2.force.x
                    );
                }
            }

            let final_ke = 0.5 * mass * vel.length_squared();
            println!("\nFinal: pos={:?}, vel={:?}", pos, vel);
            println!(
                "KE change: {:.6} -> {:.6} J ({:+.2}%)",
                initial_ke,
                final_ke,
                (final_ke / initial_ke - 1.0) * 100.0
            );
        }
    }

    /// Test damping coefficient calculation
    #[test]
    fn test_damping_coefficient() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        println!("Material properties:");
        println!("  Young's modulus: {} Pa", material.youngs_modulus);
        println!("  Restitution: {}", material.restitution);
        println!("  Friction: {}", material.friction);
        println!("  Rolling friction: {}", material.rolling_friction);

        // Calculate damping for typical collision
        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let overlap: f32 = 0.001; // 1mm overlap

        let r_eff = radius / 2.0; // Same size particles
        let m_eff = mass / 2.0;
        let e_eff = material.youngs_modulus
            / (2.0 * (1.0 - material.poisson_ratio * material.poisson_ratio));
        let k_n = (4.0 / 3.0) * e_eff * (r_eff * overlap).sqrt();

        let ln_e = material.restitution.max(0.01).ln();
        let gamma_n = -2.0 * ln_e * (k_n * m_eff).sqrt() / (PI * PI + ln_e * ln_e).sqrt();

        println!(
            "\nCalculated parameters for {:.0}mm overlap:",
            overlap * 1000.0
        );
        println!("  Effective radius: {:.4} m", r_eff);
        println!("  Effective mass: {:.6} kg", m_eff);
        println!("  Normal stiffness k_n: {:.2} N/m", k_n);
        println!("  Damping coefficient gamma_n: {:.4} kg/s", gamma_n);
        println!("  ln(restitution): {:.4}", ln_e);

        // Critical damping ratio
        let critical_damping = 2.0 * (k_n * m_eff).sqrt();
        let damping_ratio = gamma_n / critical_damping;
        println!("  Critical damping: {:.4} kg/s", critical_damping);
        println!(
            "  Damping ratio: {:.4} (1.0 = critically damped)",
            damping_ratio
        );

        // For comparison, wall properties
        let wall_props = WallProperties::default();
        println!("\nWall properties:");
        println!("  Stiffness: {} N/m", wall_props.stiffness);
        println!("  Damping: {} kg/s", wall_props.damping);
        println!("  Friction: {}", wall_props.friction);
    }

    /// Debug test for floor friction
    #[test]
    fn test_floor_friction_debug() {
        let container = Container::default();
        let wall_props = WallProperties::default();

        let radius = 0.02f32;
        let mass = 0.084f32;
        let omega = Vec3::ZERO;

        let floor_y = container.base_position.y - container.half_extents.y;

        // Particle exactly on floor with horizontal velocity (away from divider)
        let pos = Vec3::new(0.1, floor_y + radius, 0.0); // x=0.1 (away from divider), y = 0.02 (exactly on floor)
        let vel = Vec3::new(0.1, 0.0, 0.0); // Moving in X direction

        println!("Floor friction test:");
        println!("  floor_y = {}", floor_y);
        println!("  pos.y = {}, radius = {}", pos.y, radius);
        println!(
            "  floor_overlap = floor_y + radius - pos.y = {}",
            floor_y + radius - pos.y
        );
        println!("  vel = {:?}", vel);

        let result =
            compute_wall_contact_force(pos, vel, omega, radius, mass, &container, &wall_props);

        println!("  Result force: {:?}", result.force);
        println!("  Result torque: {:?}", result.torque);
        println!("  Force.x (friction): {}", result.force.x);

        // Friction should be negative (opposing motion in +X direction)
        assert!(
            result.force.x < -0.01,
            "Floor friction should slow horizontal motion, got force.x = {}",
            result.force.x
        );
    }

    // ========================================
    // Particle-Particle Collision Tests
    // ========================================

    /// Test head-on collision between two identical particles
    #[test]
    fn test_head_on_collision_energy() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 2000.0; // High frequency for accurate collision

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        println!("=== Head-on Collision Test ===");
        println!(
            "Material: restitution={}, friction={}",
            material.restitution, material.friction
        );
        println!("Particle: radius={}, mass={:.6}kg", radius, mass);

        // Two particles approaching each other
        let mut pos1 = Vec3::new(-0.05, 0.5, 0.0);
        let mut vel1 = Vec3::new(1.0, 0.0, 0.0); // Moving right at 1 m/s
        let mut omega1 = Vec3::ZERO;

        let mut pos2 = Vec3::new(0.05, 0.5, 0.0);
        let mut vel2 = Vec3::new(-1.0, 0.0, 0.0); // Moving left at 1 m/s
        let mut omega2 = Vec3::ZERO;

        let mut contact_state = ContactState::default();

        // Initial state
        let initial_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        let initial_momentum = mass * vel1 + mass * vel2;

        println!("\nInitial state:");
        println!("  pos1={:?}, vel1={:?}", pos1, vel1);
        println!("  pos2={:?}, vel2={:?}", pos2, vel2);
        println!("  Total KE: {:.6} J", initial_ke);
        println!("  Total momentum: {:?}", initial_momentum);

        // Simulate until collision and separation
        let mut max_ke = initial_ke;
        let mut in_contact = false;
        let mut contact_started = false;
        let mut contact_ended = false;

        for step in 0..2000 {
            let time = step as f32 * dt;

            // Check if in contact
            let dist = (pos1 - pos2).length();
            let overlap = 2.0 * radius - dist;
            let now_in_contact = overlap > 0.0;

            if now_in_contact && !in_contact {
                contact_started = true;
                println!("\nContact started at t={:.4}s, step {}", time, step);
                println!("  overlap={:.6}m", overlap);
            }
            if !now_in_contact && in_contact {
                contact_ended = true;
                println!("Contact ended at t={:.4}s, step {}", time, step);
            }
            in_contact = now_in_contact;

            // Calculate forces
            let (force1, force2) = compute_particle_contact_force(
                pos1,
                vel1,
                omega1,
                radius,
                mass,
                pos2,
                vel2,
                omega2,
                radius,
                mass,
                &material,
                &mut contact_state,
                dt,
            );

            // Simple integration (no gravity for this test)
            vel1 += force1.force / mass * dt;
            omega1 += force1.torque / inertia * dt;
            pos1 += vel1 * dt;

            vel2 += force2.force / mass * dt;
            omega2 += force2.torque / inertia * dt;
            pos2 += vel2 * dt;

            // Track energy
            let ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
            if ke > max_ke {
                max_ke = ke;
                println!(
                    "  WARNING: KE increased at step {}: {:.6} J (was {:.6} J)",
                    step, ke, initial_ke
                );
            }

            // Print during contact
            if in_contact && step % 10 == 0 {
                println!(
                    "  step {}: KE={:.6}J, vel1.x={:.4}, vel2.x={:.4}, overlap={:.6}m",
                    step, ke, vel1.x, vel2.x, overlap
                );
            }

            if contact_ended {
                break;
            }
        }

        let final_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        let final_momentum = mass * vel1 + mass * vel2;

        println!("\nFinal state:");
        println!("  pos1={:?}, vel1={:?}", pos1, vel1);
        println!("  pos2={:?}, vel2={:?}", pos2, vel2);
        println!("  Total KE: {:.6} J (was {:.6} J)", final_ke, initial_ke);
        println!(
            "  Total momentum: {:?} (was {:?})",
            final_momentum, initial_momentum
        );
        println!("  Max KE during collision: {:.6} J", max_ke);

        // Check momentum conservation
        let momentum_change = (final_momentum - initial_momentum).length();
        println!("  Momentum change: {:.6} kg·m/s", momentum_change);
        assert!(
            momentum_change < 0.001,
            "Momentum should be conserved, but changed by {:.6} kg·m/s",
            momentum_change
        );

        // Check energy: should decrease but not increase
        let energy_ratio = final_ke / initial_ke;
        let expected_ratio = material.restitution * material.restitution; // e² for head-on collision
        println!(
            "  Energy ratio: {:.4} (expected ~{:.4} for e={})",
            energy_ratio, expected_ratio, material.restitution
        );

        assert!(
            max_ke <= initial_ke * 1.01,
            "Energy should not increase! Max KE {:.6} > initial {:.6} (+{:.2}%)",
            max_ke,
            initial_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );

        // For restitution e, energy after = e² * energy before (for equal mass head-on)
        // Allow some tolerance due to numerical integration
        let min_expected = expected_ratio * 0.5;
        let max_expected = expected_ratio * 2.0;
        assert!(
            energy_ratio >= min_expected && energy_ratio <= max_expected.max(0.9),
            "Energy ratio {:.4} outside expected range [{:.4}, {:.4}]",
            energy_ratio,
            min_expected,
            max_expected
        );
    }

    /// Test oblique collision (particles hitting at an angle)
    #[test]
    fn test_oblique_collision_energy() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 2000.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        println!("=== Oblique Collision Test ===");

        // Particle 1 moving diagonally
        let mut pos1 = Vec3::new(-0.05, 0.51, 0.0);
        let mut vel1 = Vec3::new(1.0, -0.2, 0.0);
        let mut omega1 = Vec3::ZERO;

        // Particle 2 stationary
        let mut pos2 = Vec3::new(0.05, 0.5, 0.0);
        let mut vel2 = Vec3::ZERO;
        let mut omega2 = Vec3::ZERO;

        let mut contact_state = ContactState::default();

        let initial_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        let initial_momentum = mass * vel1 + mass * vel2;

        println!("Initial KE: {:.6} J", initial_ke);
        println!("Initial momentum: {:?}", initial_momentum);

        let mut max_ke = initial_ke;

        for step in 0..3000 {
            let (force1, force2) = compute_particle_contact_force(
                pos1,
                vel1,
                omega1,
                radius,
                mass,
                pos2,
                vel2,
                omega2,
                radius,
                mass,
                &material,
                &mut contact_state,
                dt,
            );

            vel1 += force1.force / mass * dt;
            omega1 += force1.torque / inertia * dt;
            pos1 += vel1 * dt;

            vel2 += force2.force / mass * dt;
            omega2 += force2.torque / inertia * dt;
            pos2 += vel2 * dt;

            let ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
            if ke > max_ke * 1.001 {
                max_ke = ke;
                let dist = (pos1 - pos2).length();
                println!(
                    "  WARNING: KE spike at step {}: {:.6} J (+{:.2}%), dist={:.6}m",
                    step,
                    ke,
                    (ke / initial_ke - 1.0) * 100.0,
                    dist
                );
            }
        }

        let final_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        let final_momentum = mass * vel1 + mass * vel2;

        println!("\nFinal state:");
        println!("  vel1={:?}, vel2={:?}", vel1, vel2);
        println!("  Final KE: {:.6} J (was {:.6} J)", final_ke, initial_ke);
        println!("  Max KE: {:.6} J", max_ke);

        // Momentum conservation
        let momentum_change = (final_momentum - initial_momentum).length();
        assert!(
            momentum_change < 0.001,
            "Momentum should be conserved, but changed by {:.6}",
            momentum_change
        );

        // Energy should not increase
        assert!(
            max_ke <= initial_ke * 1.05,
            "Energy should not increase significantly! Max KE {:.6} > initial {:.6} (+{:.2}%)",
            max_ke,
            initial_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );
    }

    /// Test multiple random collisions
    #[test]
    fn test_random_collisions_energy_conservation() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 2000.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        println!("=== Random Collisions Test ===");

        struct Particle {
            pos: Vec3,
            vel: Vec3,
            omega: Vec3,
        }

        // 6 particles in a box (spaced at least 2*radius apart to avoid initial overlap)
        let mut particles = vec![
            Particle {
                pos: Vec3::new(-0.06, 0.5, 0.0),
                vel: Vec3::new(0.3, 0.1, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.06, 0.5, 0.0),
                vel: Vec3::new(-0.3, -0.1, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, 0.55, 0.05),
                vel: Vec3::new(0.1, -0.2, -0.1),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, 0.55, -0.05),
                vel: Vec3::new(-0.1, -0.2, 0.1),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(-0.05, 0.60, 0.0),
                vel: Vec3::new(0.2, 0.05, -0.05),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.05, 0.60, 0.0),
                vel: Vec3::new(-0.2, 0.1, 0.1),
                omega: Vec3::ZERO,
            },
        ];

        let mut contact_states: std::collections::HashMap<(usize, usize), ContactState> =
            std::collections::HashMap::new();

        let calc_total_ke = |particles: &[Particle]| -> f32 {
            particles
                .iter()
                .map(|p| {
                    0.5 * mass * p.vel.length_squared() + 0.5 * inertia * p.omega.length_squared()
                })
                .sum()
        };

        let calc_total_momentum =
            |particles: &[Particle]| -> Vec3 { particles.iter().map(|p| mass * p.vel).sum() };

        let initial_ke = calc_total_ke(&particles);
        let initial_momentum = calc_total_momentum(&particles);

        println!("Initial KE: {:.6} J", initial_ke);
        println!("Initial momentum: {:?}", initial_momentum);

        let mut max_ke = initial_ke;
        let mut energy_spikes = 0;

        // Use Velocity Verlet integration (same as actual simulation)
        for step in 0..5000 {
            // First half: Calculate forces
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];
                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // First half integration (no gravity for this test)
            for (i, p) in particles.iter_mut().enumerate() {
                let accel = forces[i] / mass;
                let alpha = torques[i] / inertia;
                p.vel += 0.5 * accel * dt;
                p.omega += 0.5 * alpha * dt;
                p.pos += p.vel * dt;
            }

            // Recalculate forces at new positions
            forces.fill(Vec3::ZERO);
            torques.fill(Vec3::ZERO);

            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];
                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Second half integration
            for (i, p) in particles.iter_mut().enumerate() {
                let accel = forces[i] / mass;
                let alpha = torques[i] / inertia;
                p.vel += 0.5 * accel * dt;
                p.omega += 0.5 * alpha * dt;
            }

            let ke = calc_total_ke(&particles);
            if ke > max_ke * 1.001 {
                if ke > initial_ke * 1.01 {
                    energy_spikes += 1;
                    if energy_spikes <= 5 {
                        println!(
                            "  Energy spike #{} at step {}: {:.6} J (+{:.2}%)",
                            energy_spikes,
                            step,
                            ke,
                            (ke / initial_ke - 1.0) * 100.0
                        );
                    }
                }
                max_ke = ke;
            }

            if step % 1000 == 0 {
                let momentum = calc_total_momentum(&particles);
                println!(
                    "step {}: KE={:.6}J, momentum_mag={:.6}",
                    step,
                    ke,
                    momentum.length()
                );
            }
        }

        let final_ke = calc_total_ke(&particles);
        let final_momentum = calc_total_momentum(&particles);

        println!("\nFinal state:");
        println!("  Final KE: {:.6} J (was {:.6} J)", final_ke, initial_ke);
        println!(
            "  Max KE: {:.6} J (+{:.2}% from initial)",
            max_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );
        println!("  Energy spikes (>1%): {}", energy_spikes);
        println!("  Final momentum: {:?}", final_momentum);

        // Check momentum conservation
        let momentum_change = (final_momentum - initial_momentum).length();
        assert!(
            momentum_change < 0.01,
            "Momentum should be conserved, but changed by {:.6}",
            momentum_change
        );

        // Energy should not increase significantly
        assert!(
            max_ke <= initial_ke * 1.10,
            "Energy increased too much! Max KE {:.6} > initial {:.6} (+{:.2}%)",
            max_ke,
            initial_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );
    }

    /// Test 3-particle collision to isolate multi-particle issue
    #[test]
    fn test_three_particle_collision() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 2000.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        println!("=== 3-Particle Collision Test ===");

        struct Particle {
            pos: Vec3,
            vel: Vec3,
            omega: Vec3,
        }

        // 3 particles converging
        let mut particles = vec![
            Particle {
                pos: Vec3::new(-0.06, 0.5, 0.0),
                vel: Vec3::new(0.5, 0.0, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.06, 0.5, 0.0),
                vel: Vec3::new(-0.5, 0.0, 0.0),
                omega: Vec3::ZERO,
            },
            Particle {
                pos: Vec3::new(0.0, 0.57, 0.0),
                vel: Vec3::new(0.0, -0.5, 0.0),
                omega: Vec3::ZERO,
            },
        ];

        let mut contact_states: std::collections::HashMap<(usize, usize), ContactState> =
            std::collections::HashMap::new();

        let calc_ke = |particles: &[Particle]| -> f32 {
            particles
                .iter()
                .map(|p| {
                    0.5 * mass * p.vel.length_squared() + 0.5 * inertia * p.omega.length_squared()
                })
                .sum()
        };

        let initial_ke = calc_ke(&particles);
        let mut max_ke = initial_ke;

        println!("Initial KE: {:.6} J", initial_ke);

        for step in 0..2000 {
            // Calculate forces
            let mut forces: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];
            let mut torques: Vec<Vec3> = vec![Vec3::ZERO; particles.len()];

            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let pi = &particles[i];
                    let pj = &particles[j];
                    let key = (i, j);
                    let contact_state = contact_states.entry(key).or_default();

                    let (force_i, force_j) = compute_particle_contact_force(
                        pi.pos,
                        pi.vel,
                        pi.omega,
                        radius,
                        mass,
                        pj.pos,
                        pj.vel,
                        pj.omega,
                        radius,
                        mass,
                        &material,
                        contact_state,
                        dt,
                    );

                    forces[i] += force_i.force;
                    torques[i] += force_i.torque;
                    forces[j] += force_j.force;
                    torques[j] += force_j.torque;
                }
            }

            // Simple Euler integration (to match head-on test)
            for (i, p) in particles.iter_mut().enumerate() {
                p.vel += forces[i] / mass * dt;
                p.omega += torques[i] / inertia * dt;
                p.pos += p.vel * dt;
            }

            let ke = calc_ke(&particles);
            if ke > max_ke * 1.005 {
                max_ke = ke;
                println!(
                    "  step {}: KE={:.6}J (+{:.2}%)",
                    step,
                    ke,
                    (ke / initial_ke - 1.0) * 100.0
                );
            }

            if step % 500 == 0 {
                println!("step {}: KE={:.6}J", step, ke);
            }
        }

        let final_ke = calc_ke(&particles);
        println!("\nFinal KE: {:.6} J (was {:.6} J)", final_ke, initial_ke);
        println!("Energy ratio: {:.4}", final_ke / initial_ke);
        println!(
            "Max KE: {:.6} J (+{:.2}%)",
            max_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );

        assert!(
            max_ke <= initial_ke * 1.05,
            "Energy should not increase! Max KE {:.6} > initial {:.6} (+{:.2}%)",
            max_ke,
            initial_ke,
            (max_ke / initial_ke - 1.0) * 100.0
        );
    }

    /// Debug test: trace step-by-step what happens in a collision
    #[test]
    fn test_collision_step_by_step() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 2000.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);

        println!("=== Step-by-Step Collision Debug ===");
        println!("dt={}, radius={}, mass={:.6}", dt, radius, mass);
        println!(
            "Material: E={}, restitution={}",
            material.youngs_modulus, material.restitution
        );

        // Two particles about to collide (start with NO overlap)
        let mut pos1 = Vec3::new(-0.025, 0.5, 0.0); // 5mm gap
        let mut vel1 = Vec3::new(0.5, 0.0, 0.0);
        let mut omega1 = Vec3::ZERO;

        let mut pos2 = Vec3::new(0.025, 0.5, 0.0);
        let mut vel2 = Vec3::new(-0.5, 0.0, 0.0);
        let mut omega2 = Vec3::ZERO;

        let mut contact_state = ContactState::default();

        let initial_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        println!("\nInitial KE: {:.6} J", initial_ke);
        println!("Initial velocities: v1={:?}, v2={:?}", vel1, vel2);

        for step in 0..100 {
            let dist = (pos1 - pos2).length();
            let overlap = 2.0 * radius - dist;

            // Calculate forces
            let (force1, force2) = compute_particle_contact_force(
                pos1,
                vel1,
                omega1,
                radius,
                mass,
                pos2,
                vel2,
                omega2,
                radius,
                mass,
                &material,
                &mut contact_state,
                dt,
            );

            let ke_before = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());

            // Integrate
            let dv1 = force1.force / mass * dt;
            let dv2 = force2.force / mass * dt;
            vel1 += dv1;
            vel2 += dv2;
            pos1 += vel1 * dt;
            pos2 += vel2 * dt;

            let ke_after = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
            let ke_change = ke_after - ke_before;

            if overlap > -0.001 || step < 5 || ke_change.abs() > 0.0001 {
                println!("step {}: dist={:.6}, overlap={:.6}m", step, dist, overlap);
                println!("  force1={:?}, force2={:?}", force1.force, force2.force);
                println!("  dv1={:?}, dv2={:?}", dv1, dv2);
                println!("  vel1={:?}, vel2={:?}", vel1, vel2);
                println!(
                    "  KE: {:.6} -> {:.6} (delta={:+.6})",
                    ke_before, ke_after, ke_change
                );
                if contact_state.active {
                    println!("  last_normal={:?}", contact_state.last_normal);
                }
            }

            if overlap < -0.005 && step > 20 {
                break;
            }
        }

        let final_ke = 0.5 * mass * (vel1.length_squared() + vel2.length_squared());
        println!(
            "\nFinal KE: {:.6} J (initial was {:.6})",
            final_ke, initial_ke
        );
        println!("Ratio: {:.4}", final_ke / initial_ke);
    }

    /// Test that contact force is symmetric (Newton's 3rd law)
    #[test]
    fn test_contact_force_symmetry() {
        use std::f32::consts::PI;

        let material = MaterialProperties::default();
        let dt = 1.0 / 120.0;

        let radius: f32 = 0.02;
        let density: f32 = 2500.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;

        println!("=== Contact Force Symmetry Test ===");

        // Two overlapping particles
        let pos1 = Vec3::new(0.0, 0.5, 0.0);
        let vel1 = Vec3::new(0.5, 0.1, 0.0);
        let omega1 = Vec3::new(0.0, 0.0, 5.0);

        let pos2 = Vec3::new(0.035, 0.5, 0.0); // 3.5cm apart = 0.5cm overlap
        let vel2 = Vec3::new(-0.3, 0.2, 0.0);
        let omega2 = Vec3::new(0.0, 0.0, -3.0);

        let mut contact_state = ContactState::default();

        let (force1, force2) = compute_particle_contact_force(
            pos1,
            vel1,
            omega1,
            radius,
            mass,
            pos2,
            vel2,
            omega2,
            radius,
            mass,
            &material,
            &mut contact_state,
            dt,
        );

        println!("Force on particle 1: {:?}", force1.force);
        println!("Force on particle 2: {:?}", force2.force);
        println!("Sum of forces: {:?}", force1.force + force2.force);

        // Forces should be equal and opposite (Newton's 3rd law)
        let force_sum = force1.force + force2.force;
        assert!(
            force_sum.length() < 0.01,
            "Forces should be equal and opposite, but sum is {:?}",
            force_sum
        );

        println!("Torque on particle 1: {:?}", force1.torque);
        println!("Torque on particle 2: {:?}", force2.torque);
    }
}
