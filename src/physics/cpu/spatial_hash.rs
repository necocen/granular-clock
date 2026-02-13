use bevy::prelude::*;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::simulation::constants::{GridSettings, SimulationConstants};

/// 空間ハッシュグリッドによる高速な近傍検索（CPU用）
#[derive(Resource)]
pub struct SpatialHashGrid {
    pub cell_size: f32,
    pub buckets: Vec<Mutex<Vec<usize>>>,
    pub table_size: usize,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32, table_size: usize) -> Self {
        // table_sizeは2のべき乗であることを推奨
        let buckets = (0..table_size).map(|_| Mutex::new(Vec::new())).collect();
        Self {
            cell_size,
            buckets,
            table_size,
        }
    }

    /// 位置からハッシュ値を計算
    pub fn hash(&self, pos: Vec3) -> usize {
        let ix = (pos.x / self.cell_size).floor() as i32;
        let iy = (pos.y / self.cell_size).floor() as i32;
        let iz = (pos.z / self.cell_size).floor() as i32;

        // 大きな素数を使ったハッシュ関数
        let h =
            (ix.wrapping_mul(73856093)) ^ (iy.wrapping_mul(19349663)) ^ (iz.wrapping_mul(83492791));

        (h as usize) & (self.table_size - 1)
    }

    /// セルインデックスからハッシュ値を計算
    #[allow(dead_code)]
    pub fn hash_cell(&self, ix: i32, iy: i32, iz: i32) -> usize {
        let h =
            (ix.wrapping_mul(73856093)) ^ (iy.wrapping_mul(19349663)) ^ (iz.wrapping_mul(83492791));

        (h as usize) & (self.table_size - 1)
    }

    /// 位置からセルインデックスを取得
    #[allow(dead_code)]
    pub fn cell_index(&self, pos: Vec3) -> (i32, i32, i32) {
        let ix = (pos.x / self.cell_size).floor() as i32;
        let iy = (pos.y / self.cell_size).floor() as i32;
        let iz = (pos.z / self.cell_size).floor() as i32;
        (ix, iy, iz)
    }

    /// グリッドをクリア（並列処理）
    pub fn clear(&self) {
        self.buckets.par_iter().for_each(|bucket| {
            bucket.lock().clear();
        });
    }

    /// パーティクルインデックスを挿入
    pub fn insert(&self, index: usize, pos: Vec3) {
        let hash = self.hash(pos);
        self.buckets[hash].lock().push(index);
    }

    pub fn neighbor_offsets() -> &'static [(i32, i32, i32); 27] {
        static OFFSETS: [(i32, i32, i32); 27] = [
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, 1, 0),
            (-1, 1, 1),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ];
        &OFFSETS
    }
}

/// GridSettings から SpatialHashGrid を初期化するシステム
pub fn init_spatial_hash_grid(mut commands: Commands, constants: Res<SimulationConstants>) {
    let grid_settings: GridSettings = constants.grid;
    commands.insert_resource(SpatialHashGrid::new(
        grid_settings.cell_size,
        grid_settings.table_size,
    ));
}
