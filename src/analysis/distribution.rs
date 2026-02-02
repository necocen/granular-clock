use bevy::prelude::*;
use std::collections::VecDeque;

use crate::physics::{ParticleSize, Position};
use crate::simulation::Container;

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
}

impl Default for DistributionHistory {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl DistributionHistory {
    pub fn new(max_samples: usize) -> Self {
        Self {
            timestamps: VecDeque::with_capacity(max_samples),
            left_large_ratio: VecDeque::with_capacity(max_samples),
            left_small_ratio: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn clear(&mut self) {
        self.timestamps.clear();
        self.left_large_ratio.clear();
        self.left_small_ratio.clear();
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
    particles: Query<(&Position, &ParticleSize)>,
    container: Res<Container>,
    mut current: ResMut<CurrentDistribution>,
    mut history: ResMut<DistributionHistory>,
    sim_time: Res<crate::simulation::SimulationTime>,
) {
    // カウントをリセット
    current.left_large = 0;
    current.right_large = 0;
    current.left_small = 0;
    current.right_small = 0;

    // 仕切りのX座標（中央 = 0）
    let divider_x = container.base_position.x;

    for (pos, size) in particles.iter() {
        let is_left = pos.0.x < divider_x;
        match (size, is_left) {
            (ParticleSize::Large, true) => current.left_large += 1,
            (ParticleSize::Large, false) => current.right_large += 1,
            (ParticleSize::Small, true) => current.left_small += 1,
            (ParticleSize::Small, false) => current.right_small += 1,
        }
    }

    // 履歴を更新
    let total_large = current.total_large();
    let total_small = current.total_small();

    if total_large > 0 || total_small > 0 {
        history.timestamps.push_back(sim_time.elapsed);
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
