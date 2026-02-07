use bevy::prelude::*;

/// シミュレーション用のコンテナ（箱）ジオメトリ
#[derive(Resource, Clone)]
pub struct Container {
    /// 箱の半分のサイズ
    pub half_extents: Vec3,
    /// 仕切りの高さ
    pub divider_height: f32,
    /// 仕切りの厚さ
    pub divider_thickness: f32,
    /// 箱の基準位置
    pub base_position: Vec3,
    /// 現在の振動オフセット
    pub current_offset: f32,
}

impl Default for Container {
    fn default() -> Self {
        Self {
            half_extents: Vec3::new(0.2, 0.25, 0.1), // 40x50x20 cm（大幅に拡大）
            divider_height: 0.2,                   // 20 cm
            divider_thickness: 0.03,                // 3 cm
            base_position: Vec3::new(0.0, 0.075, 0.0), // 中心がY=0.075m
            current_offset: 0.0,
        }
    }
}

impl Container {
    /// コンテナの底面のY座標
    pub fn floor_y(&self) -> f32 {
        self.base_position.y - self.half_extents.y + self.current_offset
    }

    /// コンテナの天井のY座標
    pub fn ceiling_y(&self) -> f32 {
        self.base_position.y + self.half_extents.y + self.current_offset
    }

    /// 仕切りの上端のY座標
    pub fn divider_top_y(&self) -> f32 {
        self.floor_y() + self.divider_height
    }
}
