use crate::simulation::constants::{ContainerParams, WallProperties};
use mikage::glam::Vec3;

#[derive(Debug, Clone, Copy, Default)]
pub struct WallContactForce {
    pub force: Vec3,
    pub torque: Vec3,
}

fn normal_force(overlap: f32, normal_velocity: f32, mass: f32, wall: &WallProperties) -> f32 {
    let log_e = wall.restitution.max(0.01).ln();
    let ratio = -log_e / (std::f32::consts::PI.powi(2) + log_e * log_e).sqrt();
    let damping = 2.0 * ratio * (wall.stiffness * mass).sqrt();
    (wall.stiffness * overlap - damping * normal_velocity).max(0.0)
}

fn contact(
    normal: Vec3,
    overlap: f32,
    velocity: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    wall: &WallProperties,
) -> WallContactForce {
    if overlap <= 0.0 {
        return WallContactForce::default();
    }
    let arm = -radius * normal;
    let normal_velocity = velocity.dot(normal);
    let force_n = normal_force(overlap, normal_velocity, mass, wall);
    let surface_velocity = velocity + omega.cross(arm);
    let tangent = surface_velocity - surface_velocity.dot(normal) * normal;
    let speed = tangent.length();
    let friction = if speed > 1e-8 {
        -tangent / speed * (wall.friction * force_n).min(speed * wall.damping)
    } else {
        Vec3::ZERO
    };
    WallContactForce {
        force: force_n * normal + friction,
        torque: arm.cross(friction),
    }
}

/// Sphere against the finite divider, including its top and top edges.
/// Return the outward normal and penetration depth. An interior center uses
/// the nearest face; the tie break is mirror symmetric except at x=0 itself.
pub fn divider_contact(pos: Vec3, radius: f32, min: Vec3, max: Vec3) -> Option<(Vec3, f32)> {
    let delta = pos - pos.clamp(min, max);
    let distance = delta.length();
    if distance > 1e-8 {
        return (distance < radius).then_some((delta / distance, radius - distance));
    }
    let distances = [
        pos.x - min.x,
        max.x - pos.x,
        pos.y - min.y,
        max.y - pos.y,
        pos.z - min.z,
        max.z - pos.z,
    ];
    let normals = [-Vec3::X, Vec3::X, -Vec3::Y, Vec3::Y, -Vec3::Z, Vec3::Z];
    let mut nearest = if pos.x >= 0.0 { 1 } else { 0 };
    for i in 0..6 {
        if distances[i] < distances[nearest] {
            nearest = i;
        }
    }
    Some((normals[nearest], radius + distances[nearest]))
}

#[allow(clippy::too_many_arguments)]
pub fn compute_moving_wall_contact_force(
    pos: Vec3,
    vel: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    container: &ContainerParams,
    offset: f32,
    wall_velocity: f32,
    wall: &WallProperties,
) -> WallContactForce {
    let center = container.base_position + Vec3::Y * offset;
    let min = center - container.half_extents;
    let max = center + container.half_extents;
    let relative = vel - Vec3::Y * wall_velocity;
    let contacts = [
        (Vec3::X, min.x + radius - pos.x),
        (-Vec3::X, pos.x + radius - max.x),
        (Vec3::Y, min.y + radius - pos.y),
        (-Vec3::Y, pos.y + radius - max.y),
        (Vec3::Z, min.z + radius - pos.z),
        (-Vec3::Z, pos.z + radius - max.z),
    ];
    let mut total = WallContactForce::default();
    for (n, penetration) in contacts {
        let f = contact(n, penetration, relative, omega, radius, mass, wall);
        total.force += f.force;
        total.torque += f.torque;
    }
    if container.divider_height > 0.0 {
        let divider_min = Vec3::new(center.x - container.divider_thickness * 0.5, min.y, min.z);
        let divider_max = Vec3::new(
            center.x + container.divider_thickness * 0.5,
            min.y + container.divider_height,
            max.z,
        );
        if let Some((n, penetration)) = divider_contact(pos, radius, divider_min, divider_max) {
            let f = contact(n, penetration, relative, omega, radius, mass, wall);
            total.force += f.force;
            total.torque += f.torque;
        }
    }
    total
}

/// Stationary-wall convenience wrapper for contact regression tests.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn compute_wall_contact_force(
    pos: Vec3,
    vel: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    container: &ContainerParams,
    offset: f32,
    wall: &WallProperties,
) -> WallContactForce {
    compute_moving_wall_contact_force(pos, vel, omega, radius, mass, container, offset, 0.0, wall)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moving_contact_is_galilean_invariant() {
        let c = ContainerParams::default();
        let w = WallProperties::default();
        let pos = Vec3::new(0.1, -c.half_extents.y + 0.009, 0.0);
        let a = compute_moving_wall_contact_force(
            pos,
            Vec3::new(0.2, -0.3, 0.0),
            Vec3::Z,
            0.01,
            0.01,
            &c,
            0.0,
            0.0,
            &w,
        );
        let b = compute_moving_wall_contact_force(
            pos,
            Vec3::new(0.2, 1.7, 0.0),
            Vec3::Z,
            0.01,
            0.01,
            &c,
            0.0,
            2.0,
            &w,
        );
        assert!(a.force.distance(b.force) < 1e-5);
        assert!(a.torque.distance(b.torque) < 1e-6);
    }
    #[test]
    fn divider_top_and_corner_have_upward_normals() {
        let min = Vec3::new(-0.005, -0.3, -0.05);
        let max = Vec3::new(0.005, -0.1, 0.05);
        let (n, d) = divider_contact(Vec3::new(0.0, -0.091, 0.0), 0.01, min, max).unwrap();
        assert_eq!(n, Vec3::Y);
        assert!((d - 0.001).abs() < 1e-7);
        let (n, _) = divider_contact(Vec3::new(0.01, -0.095, 0.0), 0.01, min, max).unwrap();
        assert!(n.x > 0.0 && n.y > 0.0);
        assert!(divider_contact(Vec3::new(0.014, -0.091, 0.0), 0.01, min, max).is_none());
    }
}
