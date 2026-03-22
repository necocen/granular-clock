//! Creates the wgpu RenderPipeline for instanced particle rendering.

use wgpu;

/// The WGSL shader source for particle instancing.
const SHADER_SOURCE: &str = include_str!("../../assets/shaders/particle_instancing.wgsl");

/// Result of creating the particle pipeline.
pub struct ParticlePipelineResult {
    pub pipeline: wgpu::RenderPipeline,
    pub view_bind_group_layout: wgpu::BindGroupLayout,
}

/// Create the particle instanced-rendering pipeline.
///
/// - Vertex buffer 0: IcoSphere mesh (position float32x3 + normal float32x3, stride 24, Vertex step)
/// - Vertex buffer 1: InstanceData (pos_scale float32x4 + color float32x4, stride 32, Instance step)
/// - Bind group 0: SceneUniform (uniform buffer)
/// - Depth stencil: Depth32Float, depth write enabled, Less compare
pub fn create_particle_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
) -> ParticlePipelineResult {
    // Bind group layout for SceneUniform
    let view_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_view_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_pipeline_layout"),
        bind_group_layouts: &[&view_bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_instancing_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    // Vertex buffer 0: Sphere mesh (position + normal, no UV)
    let mesh_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 24, // 3 floats (position) + 3 floats (normal) = 6 * 4 = 24
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position: float32x3
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            // normal: float32x3
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ],
    };

    // Vertex buffer 1: Instance data (pos_scale + color)
    let instance_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 32, // 4 floats (pos_scale) + 4 floats (color) = 8 * 4 = 32
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            // i_pos_scale: float32x4
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            },
            // i_color: float32x4
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 3,
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("particle_render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[mesh_vertex_layout, instance_vertex_layout],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });

    ParticlePipelineResult {
        pipeline,
        view_bind_group_layout,
    }
}
