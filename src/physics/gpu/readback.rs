//! GPU → CPU 読み戻しモジュール
//!
//! GPU で計算した粒子位置を CPU に読み戻して、Main World の ParticleStore を更新する。
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

use crate::physics::ParticleStore;

/// Main World で ParticleStore を GPU 結果で更新するシステム
pub fn apply_gpu_results(
    readback: Option<Res<GpuReadbackBuffer>>,
    mut store: ResMut<ParticleStore>,
    mut last_frame: Local<u64>,
    mut debug_counter: Local<u32>,
) {
    let Some(readback) = readback else {
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

    // GPU データを read lock を保持したまま直接イテレート（clone 不要）
    let guard = readback.data.read().unwrap();

    if guard.is_empty() {
        return;
    }

    // デバッグ: 最初の数フレームだけログ出力
    *debug_counter += 1;
    let do_debug = *debug_counter <= 10;

    if do_debug {
        let mut vel_nonzero = 0;
        let mut vel_y_nonzero = 0;
        let mut pos_nan = 0;
        let mut vel_nan = 0;
        for p in guard.iter() {
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
            guard.len(),
            vel_nonzero,
            vel_y_nonzero,
            pos_nan,
            vel_nan,
        );
    }

    // ParticleStore に直接書き込み（Entity lookup 不要）
    let num = guard.len().min(store.particles.len());
    let mut actually_moved = 0;
    for i in 0..num {
        let gp = &guard[i];
        let p = &mut store.particles[i];
        let new_pos = Vec3::new(gp.pos[0], gp.pos[1], gp.pos[2]);
        if p.position != new_pos {
            actually_moved += 1;
        }
        p.position = new_pos;
        p.velocity = Vec3::new(gp.vel[0], gp.vel[1], gp.vel[2]);
        p.angular_velocity = Vec3::new(gp.omega[0], gp.omega[1], gp.omega[2]);
    }

    if do_debug {
        info!("  -> updated={}, actually_moved={}", num, actually_moved);
    }
}
