use bevy::prelude::*;

use crate::physics::{ParticleSize, Position, Velocity};

/// デバッグ用：粒子の状態を出力
#[allow(dead_code)]
pub fn debug_particles(particles: Query<(&Position, &Velocity, &ParticleSize)>, time: Res<Time>) {
    // 1秒ごとに出力
    if (time.elapsed_secs() as u32).is_multiple_of(2) {
        let mut large_count = 0;
        let mut small_count = 0;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut avg_vel = Vec3::ZERO;

        for (pos, vel, size) in particles.iter() {
            match size {
                ParticleSize::Large => large_count += 1,
                ParticleSize::Small => small_count += 1,
            }
            min_y = min_y.min(pos.0.y);
            max_y = max_y.max(pos.0.y);
            avg_vel += vel.0;
        }

        let total = large_count + small_count;
        if total > 0 {
            avg_vel /= total as f32;
            info!(
                "Particles: {} large, {} small | Y range: {:.3}..{:.3} | Avg vel: ({:.2}, {:.2}, {:.2})",
                large_count, small_count, min_y, max_y, avg_vel.x, avg_vel.y, avg_vel.z
            );
        } else {
            warn!("No particles found!");
        }
    }
}
