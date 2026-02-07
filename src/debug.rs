use bevy::prelude::*;

use crate::physics::{ParticleSize, ParticleStore};

/// デバッグ用：粒子の状態を出力
#[allow(dead_code)]
pub fn debug_particles(store: Res<ParticleStore>, time: Res<Time>) {
    // 1秒ごとに出力
    if (time.elapsed_secs() as u32).is_multiple_of(2) {
        let mut large_count = 0;
        let mut small_count = 0;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut avg_vel = Vec3::ZERO;

        for p in &store.particles {
            match p.size {
                ParticleSize::Large => large_count += 1,
                ParticleSize::Small => small_count += 1,
            }
            min_y = min_y.min(p.position.y);
            max_y = max_y.max(p.position.y);
            avg_vel += p.velocity;
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
