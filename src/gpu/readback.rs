//! GPU → CPU 読み戻しモジュール
//!
//! GPU で計算した粒子位置を CPU に読み戻して、Main World の Position コンポーネントを更新する。
//! 1フレームの遅延がある非同期読み戻し方式を使用。

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};
use std::sync::{atomic::AtomicBool, Arc, RwLock};

use super::buffers::ParticleGpu;

/// 読み戻した粒子データを保持する共有リソース（Main World と Render World で共有）
#[derive(Resource, Clone)]
pub struct GpuReadbackBuffer {
    /// 粒子データ
    pub data: Arc<RwLock<Vec<ParticleGpu>>>,
    /// フレームカウンタ（更新検知用）
    pub frame: Arc<RwLock<u64>>,
}

impl Default for GpuReadbackBuffer {
    fn default() -> Self {
        Self {
            data: Arc::new(RwLock::new(Vec::new())),
            frame: Arc::new(RwLock::new(0)),
        }
    }
}

/// Readback の頻度設定
#[derive(Resource, Clone, Copy)]
pub struct ReadbackSettings {
    /// 何フレームに1回 readback するか（1 = 毎フレーム、4 = 4フレームに1回）
    pub interval: u32,
}

impl Default for ReadbackSettings {
    fn default() -> Self {
        Self {
            interval: 4, // デフォルトは4フレームに1回
        }
    }
}

/// Render World 用のステージングバッファ
#[derive(Resource)]
pub struct ReadbackStaging {
    /// ステージングバッファ
    pub buffer: Buffer,
    /// バッファサイズ
    pub size: u64,
    /// マッピングリクエスト中か
    pub mapping_requested: bool,
    /// マッピング完了フラグ（コールバックで設定）
    pub mapping_complete: Arc<AtomicBool>,
    /// 粒子数
    pub num_particles: u32,
    /// フレームカウンタ（readback 頻度制御用）
    pub frame_counter: u32,
}

impl ReadbackStaging {
    pub fn new(render_device: &RenderDevice, num_particles: u32) -> Self {
        let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
        let size = particle_size * num_particles as u64;

        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            mapping_requested: false,
            mapping_complete: Arc::new(AtomicBool::new(false)),
            num_particles,
            frame_counter: 0,
        }
    }
}

use super::plugin::GpuParticleData;

/// Main World で Position コンポーネントを GPU 結果で更新するシステム
pub fn apply_gpu_results(
    readback: Option<Res<GpuReadbackBuffer>>,
    gpu_particle_data: Option<Res<GpuParticleData>>,
    mut particles: Query<(
        &mut crate::physics::Position,
        &mut crate::physics::Velocity,
        &mut crate::physics::AngularVelocity,
    )>,
    mut last_frame: Local<u64>,
    mut debug_counter: Local<u32>,
) {
    let Some(readback) = readback else {
        return;
    };
    let Some(gpu_particle_data) = gpu_particle_data else {
        return;
    };

    // フレームが更新されたかチェック
    let current_frame = {
        let guard = readback.frame.read().unwrap();
        *guard
    };

    if current_frame == *last_frame || current_frame == 0 {
        return; // 更新なし
    }
    *last_frame = current_frame;

    // GPU データを取得
    let gpu_data = {
        let guard = readback.data.read().unwrap();
        guard.clone()
    };

    if gpu_data.is_empty() {
        return;
    }

    // デバッグ: 最初の数フレームだけログ出力
    *debug_counter += 1;
    let do_debug = *debug_counter <= 10;

    if do_debug {
        // GPU データの統計を収集
        let mut vel_nonzero = 0;
        let mut vel_y_nonzero = 0;
        let mut pos_nan = 0;
        let mut vel_nan = 0;
        for p in gpu_data.iter() {
            if p.vel[0] != 0.0 || p.vel[1] != 0.0 || p.vel[2] != 0.0 {
                vel_nonzero += 1;
            }
            if p.vel[1] != 0.0 {
                vel_y_nonzero += 1;
            }
            if p.pos[0].is_nan() || p.pos[1].is_nan() || p.pos[2].is_nan() {
                pos_nan += 1;
            }
            if p.vel[0].is_nan() || p.vel[1].is_nan() || p.vel[2].is_nan() {
                vel_nan += 1;
            }
        }

        info!(
            "apply_gpu_results: frame={}, gpu_data={}, vel_nonzero={}, vel_y_nonzero={}, pos_nan={}, vel_nan={}",
            current_frame,
            gpu_data.len(),
            vel_nonzero,
            vel_y_nonzero,
            pos_nan,
            vel_nan,
        );

        // 最初と最後の数粒子の値をサンプル出力
        for i in [
            0usize,
            1,
            2,
            gpu_data.len() / 2,
            gpu_data.len() - 2,
            gpu_data.len() - 1,
        ] {
            if i < gpu_data.len() {
                let p = &gpu_data[i];
                info!(
                    "  particle[{}]: pos=({:.4},{:.4},{:.4}) vel=({:.4},{:.4},{:.4}) r={:.4} mass_inv={:.4}",
                    i, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2], p.radius, p.mass_inv
                );
            }
        }
    }

    // Entity リストを使って Position と Velocity を更新
    let mut updated = 0;
    let mut not_found = 0;
    let mut actually_moved = 0;
    for (i, entity) in gpu_particle_data.entities.iter().enumerate() {
        if i >= gpu_data.len() {
            break;
        }
        if let Ok((mut pos, mut vel, mut angular_vel)) = particles.get_mut(*entity) {
            let p = &gpu_data[i];
            let new_pos = Vec3::new(p.pos[0], p.pos[1], p.pos[2]);
            let new_vel = Vec3::new(p.vel[0], p.vel[1], p.vel[2]);
            let new_omega = Vec3::new(p.omega[0], p.omega[1], p.omega[2]);
            if pos.0 != new_pos {
                actually_moved += 1;
            }
            pos.0 = new_pos;
            vel.0 = new_vel;
            angular_vel.0 = new_omega;
            updated += 1;
        } else {
            not_found += 1;
        }
    }

    if do_debug {
        info!(
            "  -> updated={}, not_found={}, actually_moved={}",
            updated, not_found, actually_moved
        );
    }
}
