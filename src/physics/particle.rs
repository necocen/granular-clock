use bevy::prelude::*;
use std::f32::consts::PI;

/// 粒子の位置（物理シミュレーション用）
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Position(pub Vec3);

/// 粒子の速度
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Velocity(pub Vec3);

/// 粒子の角速度
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct AngularVelocity(pub Vec3);

/// 粒子に作用する力（毎フレームリセット）
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Force(pub Vec3);

/// 粒子に作用するトルク（毎フレームリセット）
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Torque(pub Vec3);

/// 粒子の物理的性質
#[derive(Component, Clone, Copy, Debug)]
pub struct ParticleProperties {
    pub radius: f32,
    pub mass: f32,
    /// 慣性モーメント I = 2/5 * m * r^2 (球の場合)
    pub inertia: f32,
}

impl ParticleProperties {
    pub fn new(radius: f32, density: f32) -> Self {
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let mass = density * volume;
        let inertia = (2.0 / 5.0) * mass * radius.powi(2);
        Self {
            radius,
            mass,
            inertia,
        }
    }
}

/// 粒子のサイズ区分（大/小）
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleSize {
    Large,
    Small,
}

/// 粒子エンティティのバンドル
#[derive(Bundle)]
pub struct ParticleBundle {
    pub position: Position,
    pub velocity: Velocity,
    pub angular_velocity: AngularVelocity,
    pub force: Force,
    pub torque: Torque,
    pub properties: ParticleProperties,
    pub size: ParticleSize,
}

impl ParticleBundle {
    pub fn new(position: Vec3, radius: f32, density: f32, size: ParticleSize) -> Self {
        Self {
            position: Position(position),
            velocity: Velocity(Vec3::ZERO),
            angular_velocity: AngularVelocity(Vec3::ZERO),
            force: Force(Vec3::ZERO),
            torque: Torque(Vec3::ZERO),
            properties: ParticleProperties::new(radius, density),
            size,
        }
    }
}
