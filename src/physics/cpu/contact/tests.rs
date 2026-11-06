use super::*;
use crate::physics::{Particle, ParticleSize};

fn particle(x: f32, radius: f32) -> Particle {
    Particle::new(Vec3::new(x, 0.0, 0.0), radius, 2000.0, ParticleSize::Large)
}

fn contact(
    p: &Particle,
    q: &Particle,
    material: &MaterialProperties,
) -> (ContactForce, ContactForce) {
    compute_particle_contact_force(
        p.position,
        p.velocity,
        p.angular_velocity,
        p.radius,
        p.mass,
        q.position,
        q.velocity,
        q.angular_velocity,
        q.radius,
        q.mass,
        material,
        &mut ContactState::default(),
        0.0004,
    )
}

fn assert_close(actual: Vec3, expected: Vec3) {
    let tolerance = 1e-6 * expected.length().max(1.0);
    assert!(
        actual.distance(expected) <= tolerance,
        "expected {expected:?}, got {actual:?}"
    );
}

fn contact_power(p: &Particle, q: &Particle, a: ContactForce, b: ContactForce) -> f32 {
    a.force.dot(p.velocity)
        + b.force.dot(q.velocity)
        + a.torque.dot(p.angular_velocity)
        + b.torque.dot(q.angular_velocity)
}

#[test]
fn sliding_contact_dissipates_spin_energy() {
    let material = MaterialProperties {
        rolling_friction: 0.0,
        ..Default::default()
    };
    let mut p = particle(-0.0099, 0.01);
    let mut q = particle(0.0099, 0.01);
    for (spin_p, spin_q) in [(10.0, 0.0), (0.0, 10.0), (10.0, 10.0)] {
        p.angular_velocity = Vec3::Z * spin_p;
        q.angular_velocity = Vec3::Z * spin_q;
        let (a, b) = contact(&p, &q, &material);
        assert_close(a.force, -b.force);
        assert!(a.torque.z < 0.0 && b.torque.z < 0.0);
        assert!(
            contact_power(&p, &q, a, b) < 0.0,
            "sliding friction must dissipate rotational energy"
        );
    }
}

#[test]
fn rolling_without_slip_has_no_sliding_friction() {
    let material = MaterialProperties {
        rolling_friction: 0.0,
        ..Default::default()
    };
    let mut p = particle(-0.0074, 0.01);
    let mut q = particle(0.0074, 0.005);
    // The two surface points both move at +0.1 m/s along Y.
    p.angular_velocity = Vec3::Z * 10.0;
    q.angular_velocity = -Vec3::Z * 20.0;
    let (a, b) = contact(&p, &q, &material);
    assert_eq!(a.force.y, 0.0);
    assert_eq!(b.force.y, 0.0);
    assert_close(a.torque, Vec3::ZERO);
    assert_close(b.torque, Vec3::ZERO);
}

#[test]
fn swapping_particles_preserves_each_particles_contact_force_and_torque() {
    let material = MaterialProperties::default();
    let mut p = particle(-0.0074, 0.01);
    let mut q = particle(0.0074, 0.005);
    p.velocity = Vec3::new(0.02, 0.1, -0.3);
    q.velocity = Vec3::new(-0.03, 0.2, 0.1);
    p.angular_velocity = Vec3::new(2.0, 7.0, 10.0);
    q.angular_velocity = Vec3::new(-3.0, 4.0, -8.0);
    let (a, b) = contact(&p, &q, &material);
    let (reversed_b, reversed_a) = contact(&q, &p, &material);
    assert_close(a.force, reversed_a.force);
    assert_close(b.force, reversed_b.force);
    assert_close(a.torque, reversed_a.torque);
    assert_close(b.torque, reversed_b.torque);
}

#[test]
fn mixed_size_force_limit_preserves_action_and_reaction() {
    let p = particle(-0.007, 0.01);
    let q = particle(0.007, 0.005);
    let (a, b) = contact(&p, &q, &MaterialProperties::default());
    assert_close(a.force, -b.force);
    assert!(a.force.x < 0.0 && b.force.x > 0.0);
    assert!(a.force.length() <= 1000.0 * p.mass * (1.0 + 1e-6));
    assert!(b.force.length() <= 1000.0 * q.mass * (1.0 + 1e-6));
}

#[test]
fn contact_limit_scales_forces_and_torques_together_to_preserve_dissipation() {
    let mut p = particle(-0.007, 0.01);
    let mut q = particle(0.007, 0.005);
    p.velocity = Vec3::new(0.0, 0.2, -0.1);
    q.velocity = Vec3::new(0.0, -0.3, 0.4);
    p.angular_velocity = Vec3::new(1.0, 10.0, 2.0);
    q.angular_velocity = Vec3::new(-3.0, -5.0, 1.0);
    // Exercise both force-dominated and torque-dominated limiting.
    for rolling_friction in [0.7, 20.0] {
        let soft_material = MaterialProperties {
            youngs_modulus: 1e4,
            rolling_friction,
            ..Default::default()
        };
        let stiff_material = MaterialProperties {
            youngs_modulus: 1e10,
            ..soft_material
        };
        let (soft_a, soft_b) = contact(&p, &q, &soft_material);
        let (stiff_a, stiff_b) = contact(&p, &q, &stiff_material);
        let scale = stiff_a.force.length() / soft_a.force.length();
        assert!(scale < stiff_material.youngs_modulus / soft_material.youngs_modulus);
        assert_close(stiff_a.force, soft_a.force * scale);
        assert_close(stiff_b.force, soft_b.force * scale);
        assert_close(stiff_a.torque, soft_a.torque * scale);
        assert_close(stiff_b.torque, soft_b.torque * scale);
        assert_close(stiff_a.force, -stiff_b.force);
        for (particle, force) in [(&p, stiff_a), (&q, stiff_b)] {
            assert!(force.force.length() <= 1000.0 * particle.mass * (1.0 + 1e-6));
            assert!(
                force.torque.length() <= 1000.0 * particle.mass * particle.radius * (1.0 + 1e-6)
            );
        }
        // There is no normal motion, so elastic normal forces do no work.
        assert!(contact_power(&p, &q, stiff_a, stiff_b) < 0.0);
    }
}
