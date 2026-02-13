use bevy::prelude::*;

/// コンテナの不変パラメータ
#[derive(Resource, Clone)]
pub struct ContainerParams {
    /// 箱の半分のサイズ
    pub half_extents: Vec3,
    /// 仕切りの高さ
    pub divider_height: f32,
    /// 仕切りの厚さ
    pub divider_thickness: f32,
    /// 箱の基準位置
    pub base_position: Vec3,
}

impl Default for ContainerParams {
    fn default() -> Self {
        Self {
            half_extents: Vec3::new(0.2, 0.25, 0.1),   // 40x50x20 cm
            divider_height: 0.10,                      // 10 cm
            divider_thickness: 0.02,                   // 2 cm
            base_position: Vec3::new(0.0, 0.075, 0.0), // 中心がY=0.075m
        }
    }
}
