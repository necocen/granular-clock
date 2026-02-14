use bevy::prelude::*;
use std::collections::VecDeque;

use crate::physics::{ParticleSize, ParticleStore};
use crate::simulation::{constants::SimulationConstants, state::SimulationState};

/// 粒子分布の履歴
#[derive(Resource)]
pub struct DistributionHistory {
    /// タイムスタンプ
    pub timestamps: VecDeque<f64>,
    /// 左側の大粒子の割合
    pub left_large_ratio: VecDeque<f64>,
    /// 左側の小粒子の割合
    pub left_small_ratio: VecDeque<f64>,
    /// 最大サンプル数
    pub max_samples: usize,
    /// サンプリング間隔（秒）
    pub sample_interval: f64,
    /// 最後のサンプル時刻
    pub last_sample_time: f64,
}

impl Default for DistributionHistory {
    fn default() -> Self {
        Self::new(600, 0.2) // 600サンプル x 0.2秒 = 120秒（2分）分
    }
}

impl DistributionHistory {
    pub fn new(max_samples: usize, sample_interval: f64) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(max_samples),
            left_large_ratio: VecDeque::with_capacity(max_samples),
            left_small_ratio: VecDeque::with_capacity(max_samples),
            max_samples,
            sample_interval,
            last_sample_time: -1.0,
        }
    }

    pub fn clear(&mut self) {
        self.timestamps.clear();
        self.left_large_ratio.clear();
        self.left_small_ratio.clear();
        self.last_sample_time = -1.0;
    }
}

/// 現在の粒子分布
#[derive(Resource, Default)]
pub struct CurrentDistribution {
    /// 左側の大粒子数
    pub left_large: u32,
    /// 右側の大粒子数
    pub right_large: u32,
    /// 左側の小粒子数
    pub left_small: u32,
    /// 右側の小粒子数
    pub right_small: u32,
}

impl CurrentDistribution {
    pub fn total_large(&self) -> u32 {
        self.left_large + self.right_large
    }

    pub fn total_small(&self) -> u32 {
        self.left_small + self.right_small
    }

    pub fn left_large_ratio(&self) -> f64 {
        let total = self.total_large();
        if total > 0 {
            self.left_large as f64 / total as f64
        } else {
            0.5
        }
    }

    pub fn left_small_ratio(&self) -> f64 {
        let total = self.total_small();
        if total > 0 {
            self.left_small as f64 / total as f64
        } else {
            0.5
        }
    }
}

/// 分布を更新するシステム
pub fn update_distribution(
    store: Res<ParticleStore>,
    constants: Res<SimulationConstants>,
    mut current: ResMut<CurrentDistribution>,
    mut history: ResMut<DistributionHistory>,
    sim_state: Res<SimulationState>,
) {
    // カウントをリセット
    current.left_large = 0;
    current.right_large = 0;
    current.left_small = 0;
    current.right_small = 0;

    // 仕切りのX座標（中央 = 0）
    let divider_x = constants.container.base_position.x;

    for p in &store.particles {
        let is_left = p.position.x < divider_x;
        match (p.size, is_left) {
            (ParticleSize::Large, true) => current.left_large += 1,
            (ParticleSize::Large, false) => current.right_large += 1,
            (ParticleSize::Small, true) => current.left_small += 1,
            (ParticleSize::Small, false) => current.right_small += 1,
        }
    }

    // 履歴を更新（サンプリング間隔を考慮）
    let total_large = current.total_large();
    let total_small = current.total_small();
    let elapsed = sim_state.elapsed;

    // サンプリング間隔が経過した場合のみ記録
    if (total_large > 0 || total_small > 0)
        && (elapsed - history.last_sample_time >= history.sample_interval)
    {
        history.last_sample_time = elapsed;
        history.timestamps.push_back(elapsed);
        history
            .left_large_ratio
            .push_back(current.left_large_ratio());
        history
            .left_small_ratio
            .push_back(current.left_small_ratio());

        // 古いデータを削除
        while history.timestamps.len() > history.max_samples {
            history.timestamps.pop_front();
            history.left_large_ratio.pop_front();
            history.left_small_ratio.pop_front();
        }
    }
}
