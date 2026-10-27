use bytemuck::{Pod, Zeroable};
use mikage::wgpu::{self, util::DeviceExt};

/// GPU上の粒子データ構造（16バイトアライメント）
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ParticleGpu {
    /// 位置 (12 bytes)
    pub pos: [f32; 3],
    /// 半径 (4 bytes) → 16 bytes aligned
    pub radius: f32,
    /// 速度 (12 bytes)
    pub vel: [f32; 3],
    /// 逆質量 (4 bytes) → 16 bytes aligned
    pub mass_inv: f32,
    /// 角速度 (12 bytes)
    pub omega: [f32; 3],
    /// 逆慣性モーメント (4 bytes) → 16 bytes aligned
    pub inertia_inv: f32,
    /// サイズフラグ (0=small, 1=large) (4 bytes)
    pub size_flag: u32,
    /// パディング (12 bytes) → 16 bytes aligned
    pub _pad: [u32; 3],
}

impl Default for ParticleGpu {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            radius: 0.01,
            vel: [0.0; 3],
            mass_inv: 1.0,
            omega: [0.0; 3],
            inertia_inv: 1.0,
            size_flag: 0,
            _pad: [0; 3],
        }
    }
}

/// シミュレーションパラメータ（uniform buffer用）
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SimulationParams {
    /// タイムステップ
    pub dt: f32,
    /// 重力加速度（Y軸、負の値）
    pub gravity: f32,
    /// セルサイズ
    pub cell_size: f32,
    /// グリッド次元（1軸あたり）
    pub grid_dim: u32,

    /// ワールド半分サイズ (12 bytes)
    pub world_half: [f32; 3],
    /// 粒子数
    pub num_particles: u32,

    /// ヤング率
    pub youngs_modulus: f32,
    /// ポアソン比
    pub poisson_ratio: f32,
    /// 反発係数
    pub restitution: f32,
    /// 摩擦係数
    pub friction: f32,

    /// 容器オフセット（振動）
    pub container_offset: f32,
    /// 仕切り高さ
    pub divider_height: f32,
    /// 容器 half_extents.x
    pub container_half_x: f32,
    /// 容器 half_extents.y
    pub container_half_y: f32,

    /// 容器 half_extents.z
    pub container_half_z: f32,
    /// 仕切り厚さ
    pub divider_thickness: f32,
    /// 転がり摩擦係数
    pub rolling_friction: f32,
    /// 壁との反発係数（粒子間とは別）
    pub wall_restitution: f32,
    /// 壁との摩擦係数（粒子間とは別）
    pub wall_friction: f32,
    /// 壁減衰係数
    pub wall_damping: f32,
    /// 壁剛性
    pub wall_stiffness: f32,
    /// パディング（uniformレイアウト安定化）
    pub _pad_end: f32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 120.0,
            gravity: -9.81,
            cell_size: 0.05,
            grid_dim: 32,
            world_half: [0.1, 0.15, 0.05],
            num_particles: 0,
            youngs_modulus: 1e6,
            poisson_ratio: 0.25,
            restitution: 0.5,
            friction: 0.5,
            container_offset: 0.0,
            divider_height: 0.05,
            container_half_x: 0.1,
            container_half_y: 0.15,
            container_half_z: 0.05,
            divider_thickness: 0.005,
            rolling_friction: 0.1,
            wall_restitution: 0.3,
            wall_friction: 0.4,
            wall_damping: 20.0,
            wall_stiffness: 10000.0,
            _pad_end: 0.0,
        }
    }
}

/// セル範囲（ソート後の粒子インデックス範囲）
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct CellRange {
    pub start: u32,
    pub end: u32,
}

/// Bitonic sort パラメータ（uniform buffer用、16byte）
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct SortParamsGpu {
    pub j: u32,
    pub k: u32,
    pub n: u32,
    pub _pad: u32,
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

fn bitonic_sort_pass_count(n: u32) -> u32 {
    if n < 2 {
        return 0;
    }
    let mut passes = 0u32;
    let mut k = 2u32;
    while k <= n {
        let mut j = k / 2;
        while j > 0 {
            passes += 1;
            j /= 2;
        }
        k *= 2;
    }
    passes
}

/// GPU バッファリソース
pub struct GpuPhysicsBuffers {
    /// 粒子データ A（各サブステップ後の最終結果スロット）
    pub particles_a: wgpu::Buffer,
    /// 粒子データ B（中間スロット）
    pub particles_b: wgpu::Buffer,
    /// 空間ハッシュキー
    pub keys: wgpu::Buffer,
    /// 粒子ID（ソート用）
    pub particle_ids: wgpu::Buffer,
    /// セル範囲
    pub cell_ranges: wgpu::Buffer,
    /// 力の累積
    pub forces: wgpu::Buffer,
    /// トルクの累積
    pub torques: wgpu::Buffer,
    /// シミュレーションパラメータ
    pub params: wgpu::Buffer,
    /// Bitonic sort パラメータ
    pub sort_params: wgpu::Buffer,
    /// sort_params の動的オフセット間隔（bytes）
    pub sort_params_stride: u32,
    /// sort_params に格納可能な最大パス数
    pub sort_params_capacity_passes: u32,
    /// 粒子数
    pub num_particles: u32,
    /// バッファ容量（粒子スロット数）
    pub capacity: u32,
    /// 最後にアップロードしたデータ世代
    #[allow(dead_code)]
    pub last_uploaded_generation: u64,
}

impl GpuPhysicsBuffers {
    pub fn new(device: &wgpu::Device, num_particles: u32, grid_size: u32) -> Self {
        let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
        let particles_buffer_size = particle_size * num_particles as u64;

        let particles_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_a"),
            size: particles_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let particles_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_b"),
            size: particles_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("keys"),
            size: 4 * num_particles as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_ids"),
            size: 4 * num_particles as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let num_cells = grid_size * grid_size * grid_size;
        let cell_ranges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_ranges"),
            size: std::mem::size_of::<CellRange>() as u64 * num_cells as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forces"),
            size: 16 * num_particles as u64, // vec4<f32> per particle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let torques = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torques"),
            size: 16 * num_particles as u64, // vec4<f32> per particle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation_params"),
            size: std::mem::size_of::<SimulationParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sort_params_size = std::mem::size_of::<SortParamsGpu>() as u64;
        let sort_params_alignment =
            device.limits().min_uniform_buffer_offset_alignment.max(1) as u64;
        let sort_params_stride = align_up(sort_params_size, sort_params_alignment) as u32;
        let sort_params_capacity_passes = bitonic_sort_pass_count(num_particles.max(2));
        let sort_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitonic_sort_params"),
            size: sort_params_stride as u64 * sort_params_capacity_passes.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            particles_a,
            particles_b,
            keys,
            particle_ids,
            cell_ranges,
            forces,
            torques,
            params,
            sort_params,
            sort_params_stride,
            sort_params_capacity_passes,
            num_particles,
            capacity: num_particles,
            last_uploaded_generation: 0,
        }
    }

    /// Velocity Verlet 前半の入力（A）を取得
    pub fn forward_in(&self) -> &wgpu::Buffer {
        &self.particles_a
    }

    /// Velocity Verlet 前半の出力（B）を取得
    pub fn forward_out(&self) -> &wgpu::Buffer {
        &self.particles_b
    }

    /// Velocity Verlet 後半の入力（B）を取得
    pub fn reverse_in(&self) -> &wgpu::Buffer {
        &self.particles_b
    }

    /// Velocity Verlet 後半の出力（A）を取得
    pub fn reverse_out(&self) -> &wgpu::Buffer {
        &self.particles_a
    }

    /// 最新の粒子状態（A）を取得
    pub fn latest_particles(&self) -> &wgpu::Buffer {
        &self.particles_a
    }

    /// ステージングバッファを作成（readback 用）
    pub fn create_staging_buffer(device: &wgpu::Device, num_particles: u32) -> wgpu::Buffer {
        let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
        let size = particle_size * num_particles as u64;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// substep params upload buffer を作成
    pub fn create_params_upload_buffer(
        device: &wgpu::Device,
        params_list: &[SimulationParams],
    ) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_physics_substep_params_upload"),
            contents: bytemuck::cast_slice(params_list),
            usage: wgpu::BufferUsages::COPY_SRC,
        })
    }
}
