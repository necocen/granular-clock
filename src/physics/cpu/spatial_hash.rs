use bevy::prelude::*;
use std::collections::HashMap;

use crate::simulation::constants::{GridSettings, SimulationConstants};

pub type GridCell = (i32, i32, i32);

/// 空間ハッシュグリッドによる高速な近傍検索（CPU用）
#[derive(Resource)]
pub struct SpatialHashGrid {
    pub cell_size: f32,
    pub table_size: usize,
    cells: HashMap<GridCell, Vec<usize>>,
    particle_cells: Vec<GridCell>,
}

impl Default for SpatialHashGrid {
    fn default() -> Self {
        Self::new(
            GridSettings::default().cell_size,
            GridSettings::default().table_size,
        )
    }
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32, table_size: usize) -> Self {
        Self {
            cell_size,
            table_size,
            cells: HashMap::with_capacity(table_size),
            particle_cells: Vec::new(),
        }
    }

    /// 位置からセルインデックスを取得
    pub fn cell_index(&self, pos: Vec3) -> GridCell {
        let ix = (pos.x / self.cell_size).floor() as i32;
        let iy = (pos.y / self.cell_size).floor() as i32;
        let iz = (pos.z / self.cell_size).floor() as i32;
        (ix, iy, iz)
    }

    /// 粒子位置からセルマップを再構築
    pub fn rebuild<I>(&mut self, positions: I)
    where
        I: IntoIterator<Item = (usize, Vec3)>,
    {
        self.cells.clear();
        self.particle_cells.clear();
        for (index, pos) in positions {
            let cell = self.cell_index(pos);
            if index >= self.particle_cells.len() {
                self.particle_cells.resize(index + 1, (0, 0, 0));
            }
            self.particle_cells[index] = cell;
            self.cells.entry(cell).or_default().push(index);
        }
    }

    /// 読み取り専用でセルマップにアクセス
    pub fn with_cells<R>(&self, f: impl FnOnce(&HashMap<GridCell, Vec<usize>>) -> R) -> R {
        f(&self.cells)
    }

    /// 粒子インデックスに対応するセル座標を返す（直近 rebuild 結果）
    pub fn particle_cell(&self, index: usize) -> GridCell {
        self.particle_cells[index]
    }
}

impl SpatialHashGrid {
    #[allow(dead_code)]
    /// セルインデックスからハッシュ値を計算（デバッグ/将来用途）
    pub fn hash_cell(&self, ix: i32, iy: i32, iz: i32) -> usize {
        let h =
            (ix.wrapping_mul(73856093)) ^ (iy.wrapping_mul(19349663)) ^ (iz.wrapping_mul(83492791));

        (h as usize) & (self.table_size - 1)
    }

    pub fn neighbor_offsets() -> &'static [GridCell; 27] {
        static OFFSETS: [GridCell; 27] = [
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
