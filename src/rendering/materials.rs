use bevy::prelude::*;

use crate::physics::{ParticleBundle, ParticleSize};
use crate::simulation::{Container, SimulationConfig};

/// 粒子用のメッシュとマテリアルのハンドル
#[derive(Resource)]
pub struct ParticleMeshes {
    pub sphere: Handle<Mesh>,
    pub large_material: Handle<StandardMaterial>,
    pub small_material: Handle<StandardMaterial>,
}

/// コンテナ用のメッシュエンティティ
#[derive(Resource)]
pub struct ContainerEntities {
    pub floor: Entity,
    pub divider: Entity,
}

/// レンダリング用のリソースをセットアップ
pub fn setup_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    container: Res<Container>,
) {
    // 粒子用の球メッシュ（低ポリゴン）
    let sphere_mesh = meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap());

    // 大粒子用マテリアル（赤）
    let large_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.2, 0.2),
        metallic: 0.3,
        perceptual_roughness: 0.5,
        ..default()
    });

    // 小粒子用マテリアル（青）
    let small_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 0.8),
        metallic: 0.3,
        perceptual_roughness: 0.5,
        ..default()
    });

    commands.insert_resource(ParticleMeshes {
        sphere: sphere_mesh,
        large_material,
        small_material,
    });

    // 壁用マテリアル（とても透明）
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.5, 0.5, 0.5, 0.05),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // 床用マテリアル（やや透明、でも見える）
    let floor_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.4, 0.35, 0.3, 0.4),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // 床
    let floor_mesh = meshes.add(Cuboid::new(
        container.half_extents.x * 2.0,
        0.005,
        container.half_extents.z * 2.0,
    ));
    let floor = commands
        .spawn((
            Mesh3d(floor_mesh),
            MeshMaterial3d(floor_material),
            Transform::from_translation(
                container.base_position - Vec3::Y * container.half_extents.y,
            ),
        ))
        .id();

    // 仕切り用マテリアル（やや不透明）
    let divider_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.4, 0.4, 0.4, 0.6),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    // 仕切り
    let divider_mesh = meshes.add(Cuboid::new(
        container.divider_thickness,
        container.divider_height,
        container.half_extents.z * 2.0,
    ));
    let divider = commands
        .spawn((
            Mesh3d(divider_mesh),
            MeshMaterial3d(divider_material),
            Transform::from_translation(
                container.base_position - Vec3::Y * container.half_extents.y
                    + Vec3::Y * container.divider_height / 2.0,
            ),
        ))
        .id();

    // 側面の壁（ワイヤーフレーム的に薄い壁で表現）
    let wall_thickness = 0.002;

    // 前後の壁
    let front_back_mesh = meshes.add(Cuboid::new(
        container.half_extents.x * 2.0,
        container.half_extents.y * 2.0,
        wall_thickness,
    ));

    // 前壁
    commands.spawn((
        Mesh3d(front_back_mesh.clone()),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_translation(container.base_position + Vec3::Z * container.half_extents.z),
    ));

    // 後壁
    commands.spawn((
        Mesh3d(front_back_mesh),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_translation(container.base_position - Vec3::Z * container.half_extents.z),
    ));

    // 左右の壁
    let left_right_mesh = meshes.add(Cuboid::new(
        wall_thickness,
        container.half_extents.y * 2.0,
        container.half_extents.z * 2.0,
    ));

    // 左壁
    commands.spawn((
        Mesh3d(left_right_mesh.clone()),
        MeshMaterial3d(wall_material.clone()),
        Transform::from_translation(container.base_position - Vec3::X * container.half_extents.x),
    ));

    // 右壁
    commands.spawn((
        Mesh3d(left_right_mesh),
        MeshMaterial3d(wall_material),
        Transform::from_translation(container.base_position + Vec3::X * container.half_extents.x),
    ));

    commands.insert_resource(ContainerEntities { floor, divider });
}

/// 粒子をスポーン
pub fn spawn_particles(
    mut commands: Commands,
    meshes: Res<ParticleMeshes>,
    config: Res<SimulationConfig>,
    container: Res<Container>,
) {
    use rand::Rng;
    let mut rng = rand::rng();

    let spawn_area_x = container.half_extents.x - config.large_radius;
    let spawn_area_z = container.half_extents.z - config.large_radius;
    let base_y = container.base_position.y - container.half_extents.y;

    // 大粒子をスポーン
    for _ in 0..config.num_large {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + config.large_radius + rng.random_range(0.0..0.2);

        let pos = Vec3::new(x, y, z);

        commands.spawn((
            ParticleBundle::new(
                pos,
                config.large_radius,
                config.density,
                ParticleSize::Large,
            ),
            Mesh3d(meshes.sphere.clone()),
            MeshMaterial3d(meshes.large_material.clone()),
            Transform::from_translation(pos).with_scale(Vec3::splat(config.large_radius)),
        ));
    }

    // 小粒子をスポーン
    for _ in 0..config.num_small {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + config.small_radius + rng.random_range(0.0..0.2);

        let pos = Vec3::new(x, y, z);

        commands.spawn((
            ParticleBundle::new(
                pos,
                config.small_radius,
                config.density,
                ParticleSize::Small,
            ),
            Mesh3d(meshes.sphere.clone()),
            MeshMaterial3d(meshes.small_material.clone()),
            Transform::from_translation(pos).with_scale(Vec3::splat(config.small_radius)),
        ));
    }
}

/// コンテナの位置を更新
pub fn update_container_transforms(
    container: Res<Container>,
    mut transforms: Query<&mut Transform>,
    entities: Option<Res<ContainerEntities>>,
) {
    let Some(entities) = entities else {
        return;
    };

    let offset = Vec3::Y * container.current_offset;

    // 床を更新
    if let Ok(mut transform) = transforms.get_mut(entities.floor) {
        transform.translation =
            container.base_position - Vec3::Y * container.half_extents.y + offset;
    }

    // 仕切りを更新
    if let Ok(mut transform) = transforms.get_mut(entities.divider) {
        transform.translation = container.base_position - Vec3::Y * container.half_extents.y
            + Vec3::Y * container.divider_height / 2.0
            + offset;
    }
}
