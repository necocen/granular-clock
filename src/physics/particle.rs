use bevy::prelude::*;
use std::f32::consts::PI;

/// 粒子のサイズ区分（大/小）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleSize {
    Large,
    Small,
}

/// 個別パーティクルのデータ（旧 ECS コンポーネントを統合）
#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub force: Vec3,
    pub torque: Vec3,
    pub radius: f32,
    pub mass: f32,
    pub inertia: f32,
    pub size: ParticleSize,
}

impl Particle {
    pub fn new(position: Vec3, radius: f32, density: f32, size: ParticleSize) -> Self {
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);
        Self {
            position,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            radius,
            mass,
            inertia,
            size,
        }
    }
}

/// 全パーティクルを Vec で管理するリソース
#[derive(Resource, Default)]
pub struct ParticleStore {
    pub particles: Vec<Particle>,
    /// spawn/clear 操作でインクリメントされる世代番号。
    /// GPU readback による書き戻しでは変化しない。
    pub generation: u64,
}

impl ParticleStore {
    /// パーティクルを追加
    pub fn spawn(&mut self, position: Vec3, radius: f32, density: f32, size: ParticleSize) {
        self.particles
            .push(Particle::new(position, radius, density, size));
        self.generation += 1;
    }

    /// 全パーティクルを削除
    pub fn clear(&mut self) {
        self.particles.clear();
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}
