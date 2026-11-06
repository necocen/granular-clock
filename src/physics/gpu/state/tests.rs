use super::*;
use crate::physics::{
    Particle, ParticleSize, ParticleStore,
    cpu::{ContactHistory, SpatialHashGrid, run_physics_substeps_plain},
};
use crate::simulation::state::SimulationState;
use glam::Vec3;

fn to_gpu(p: &Particle) -> ParticleGpu {
    ParticleGpu {
        pos: p.position.into(),
        radius: p.radius,
        vel: p.velocity.into(),
        mass_inv: 1.0 / p.mass,
        omega: p.angular_velocity.into(),
        inertia_inv: 1.0 / p.inertia,
        size_flag: u32::from(p.size == ParticleSize::Large),
        _pad: [0; 3],
    }
}

fn read_particles(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &GpuPhysicsState,
) -> Vec<ParticleGpu> {
    let count = gpu.num_particles();
    let size = count as u64 * std::mem::size_of::<ParticleGpu>() as u64;
    let staging = GpuPhysicsBuffers::create_staging_buffer(device, count);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(gpu.buffers.latest_particles(), 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap();
    let particles = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    particles
}

fn assert_close(label: &str, field: &str, actual: Vec3, expected: Vec3, tolerance: f32) {
    assert!(
        actual.distance(expected) <= tolerance * expected.length().max(1.0),
        "{label}: {field}: CPU {expected:?}, GPU {actual:?}"
    );
}

fn compare_steps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    constants: &SimulationConstants,
    initial: Vec<Particle>,
) {
    let initial_momentum: Vec3 = initial.iter().map(|p| p.mass * p.velocity).sum();
    let uploaded: Vec<_> = initial.iter().map(to_gpu).collect();
    let mut store = ParticleStore {
        particles: initial,
        generation: 0,
    };
    let mut history = ContactHistory::default();
    let mut grid = SpatialHashGrid::new(constants.grid.cell_size, constants.grid.table_size);
    let mut state = SimulationState::default();
    let instances = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contact_test_instances"),
        size: 4096,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut gpu = GpuPhysicsState::new(device, constants, &instances);
    gpu.upload_particles(device, queue, &uploaded, &instances);
    gpu.update_params(queue, constants, &state);
    // Exercise the production grid, sorting, collision and both Verlet half steps.
    for step in 0..10 {
        let start_phase = state.oscillation_phase;
        run_physics_substeps_plain(constants, &mut store, &mut history, &mut state, &mut grid);
        state.oscillation_frame_start_phase = start_phase;
        let mut encoder = device.create_command_encoder(&Default::default());
        gpu.encode_compute(device, queue, &mut encoder, constants, &state);
        queue.submit([encoder.finish()]);
        if step != 0 && step != 9 {
            continue;
        }
        let actual = read_particles(device, queue, &gpu);
        let mut momentum = Vec3::ZERO;
        for (p, gp) in store.particles.iter().zip(&actual) {
            assert_close(
                label,
                "position",
                Vec3::from_array(gp.pos),
                p.position,
                2e-6,
            );
            assert_close(
                label,
                "velocity",
                Vec3::from_array(gp.vel),
                p.velocity,
                2e-5,
            );
            assert_close(
                label,
                "spin",
                Vec3::from_array(gp.omega),
                p.angular_velocity,
                2e-4,
            );
            momentum += p.mass * Vec3::from_array(gp.vel);
        }
        if !constants.oscillation.enabled {
            assert_close(label, "total momentum", momentum, initial_momentum, 1e-6);
        }
    }
}

#[test]
#[ignore = "requires native GPU adapter/device and runs real compute passes"]
fn rotating_and_mixed_size_contacts_match_cpu_and_conserve_momentum() {
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU adapter required for contact regression test");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("GPU device required for contact regression test")
    });
    let mut c = crate::simulation::config_toml::load_embedded_config().simulation;
    c.oscillation.enabled = false;
    c.physics.gravity = Vec3::ZERO;
    c.container.divider_height = 0.0;
    c.settings.substeps_per_frame = 1;
    c.material.rolling_friction = 0.0;
    for dt in [1e-5, c.time.dt] {
        c.time.dt = dt;
        for (radius, spin_p, spin_q) in [
            (0.01, 10.0, 0.0),
            (0.01, 0.0, 10.0),
            (0.01, 10.0, 10.0),
            (0.005, 10.0, -20.0),
            (0.005, 0.0, 0.0),
            (0.005, 10.0, 3.0),
        ] {
            let distance = 0.01 + radius - 0.0002;
            let mut p = Particle::new(
                Vec3::new(-distance / 2.0, 0.1, 0.0),
                0.01,
                2000.0,
                ParticleSize::Large,
            );
            let mut q = Particle::new(
                Vec3::new(distance / 2.0, 0.1, 0.0),
                radius,
                2000.0,
                ParticleSize::Small,
            );
            p.angular_velocity = Vec3::Z * spin_p;
            q.angular_velocity = Vec3::Z * spin_q;
            let label = format!("dt={dt}, radius={radius}, spins={spin_p}/{spin_q}");
            compare_steps(&device, &queue, &label, &c, vec![p.clone(), q.clone()]);
            compare_steps(
                &device,
                &queue,
                &format!("reversed {label}"),
                &c,
                vec![q, p],
            );
        }
    }

    // A large rolling torque makes the smaller particle's torque bound active.
    c.material.rolling_friction = 20.0;
    let mut p = Particle::new(
        Vec3::new(-0.007, 0.1, 0.0),
        0.01,
        2000.0,
        ParticleSize::Large,
    );
    let mut q = Particle::new(
        Vec3::new(0.007, 0.1, 0.0),
        0.005,
        2000.0,
        ParticleSize::Small,
    );
    p.velocity = Vec3::new(0.0, 0.2, -0.1);
    q.velocity = Vec3::new(0.0, -0.3, 0.4);
    p.angular_velocity = Vec3::new(1.0, 10.0, 2.0);
    q.angular_velocity = Vec3::new(-3.0, -5.0, 1.0);
    compare_steps(
        &device,
        &queue,
        "torque limited",
        &c,
        vec![p.clone(), q.clone()],
    );
    compare_steps(&device, &queue, "torque limited reversed", &c, vec![q, p]);

    c.material.rolling_friction = 0.1;
    let particles = (0..8)
        .map(|i| {
            let radius = if i % 2 == 0 { 0.01 } else { 0.005 };
            let mut p = Particle::new(
                Vec3::new(
                    (i % 2) as f32 * 0.014,
                    0.1 + (i / 2 % 2) as f32 * 0.014,
                    (i / 4) as f32 * 0.014,
                ),
                radius,
                2000.0,
                ParticleSize::Large,
            );
            p.velocity = Vec3::new(0.01 * i as f32, -0.02, 0.03);
            p.angular_velocity = Vec3::new(2.0, i as f32 - 4.0, 3.0);
            p
        })
        .collect();
    compare_steps(&device, &queue, "dense contacts", &c, particles);
    c.oscillation.enabled = true;
    c.oscillation.amplitude = 0.002;
    c.oscillation.frequency = 10.0;
    c.container.divider_height = 0.17;
    let floor = -c.container.half_extents.y;
    let initial = vec![
        Particle::new(
            Vec3::new(0.1, floor + 0.0095, 0.0),
            0.01,
            2000.0,
            ParticleSize::Large,
        ),
        Particle::new(
            Vec3::new(0.0, floor + 0.1795, 0.0),
            0.01,
            2000.0,
            ParticleSize::Large,
        ),
        Particle::new(
            Vec3::new(0.01, floor + 0.175, 0.03),
            0.01,
            2000.0,
            ParticleSize::Large,
        ),
    ];
    // External moving walls exchange momentum with particles, so compare states
    // without the isolated-system momentum assertion.
    compare_steps(
        &device,
        &queue,
        "moving walls and divider edges",
        &c,
        initial,
    );
}
