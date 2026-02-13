#define_import_path granular_clock::physics_types

struct Particle {
    pos: vec3<f32>,
    radius: f32,
    vel: vec3<f32>,
    mass_inv: f32,
    omega: vec3<f32>,
    inertia_inv: f32,
    size_flag: u32,
    _pad: array<u32, 3>,
}

struct Params {
    dt: f32,
    gravity: f32,
    cell_size: f32,
    grid_dim: u32,
    world_half: vec3<f32>,
    num_particles: u32,
    youngs_modulus: f32,
    poisson_ratio: f32,
    restitution: f32,
    friction: f32,
    container_offset: f32,
    divider_height: f32,
    container_half_x: f32,
    container_half_y: f32,
    container_half_z: f32,
    divider_thickness: f32,
    rolling_friction: f32,
    wall_restitution: f32,
    wall_friction: f32,
    wall_damping: f32,
    wall_stiffness: f32,
    _pad_end: f32,
}

// 原点基準のセル座標（CPU実装と同じ）
fn grid_cell_from_pos(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(pos / cell_size));
}

// セル座標を [0, grid_dim^3) の線形インデックスへ変換
fn grid_hash_cell(cell: vec3<i32>, grid_dim: u32) -> u32 {
    let dim = i32(grid_dim);
    let half = dim / 2;
    let shifted = cell + vec3<i32>(half, half, half);
    let c = vec3<i32>(
        clamp(shifted.x, 0, dim - 1),
        clamp(shifted.y, 0, dim - 1),
        clamp(shifted.z, 0, dim - 1),
    );
    return u32((c.z * dim + c.y) * dim + c.x);
}

fn grid_num_cells(grid_dim: u32) -> u32 {
    return grid_dim * grid_dim * grid_dim;
}

fn grid_cell_in_bounds(cell: vec3<i32>, grid_dim: u32) -> bool {
    let dim = i32(grid_dim);
    let half = dim / 2;
    let min_cell = -half;
    let max_cell = min_cell + dim;

    return (
        cell.x >= min_cell && cell.x < max_cell &&
        cell.y >= min_cell && cell.y < max_cell &&
        cell.z >= min_cell && cell.z < max_cell
    );
}
