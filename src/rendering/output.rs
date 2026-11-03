//! Scene attachments and the HDR-to-display pass, with a direct-render fallback.

use mikage::dpi::PhysicalSize;
use mikage::{
    GpuContext, OffscreenTarget, RenderTarget, RenderTargetConfig, ToneMapSource, ToneMapper, wgpu,
};

pub enum SceneOutput {
    Direct(wgpu::TextureView),
    Hdr {
        target: Box<OffscreenTarget>,
        mapper: ToneMapper,
        source: ToneMapSource,
    },
}

impl SceneOutput {
    pub fn new(
        gpu: &GpuContext,
        final_config: RenderTargetConfig,
        size: PhysicalSize<u32>,
    ) -> Self {
        let hdr_config = RenderTargetConfig {
            color_format: mikage::HDR_FORMAT,
            ..final_config
        };
        let hdr = OffscreenTarget::new(gpu, size, hdr_config)
            .and_then(|target| ToneMapper::new(gpu, final_config).map(|mapper| (target, mapper)));
        match hdr {
            Ok((target, mapper)) => {
                // Fixed display exposure keeps the simulation readable.
                mapper.update_exposure(&gpu.queue, 0.5);
                let source = mapper.source(gpu, target.view());
                return Self::Hdr {
                    target: Box::new(target),
                    mapper,
                    source,
                };
            }
            Err(error) => {
                tracing::warn!("HDR output is unavailable; using direct rendering: {error}")
            }
        }
        let (_, depth) = mikage::create_depth_texture(gpu, size, final_config);
        Self::Direct(depth)
    }

    pub fn config(&self, final_config: RenderTargetConfig) -> RenderTargetConfig {
        match self {
            Self::Direct(_) => final_config,
            Self::Hdr { target, .. } => target.render_target_config(),
        }
    }

    pub fn target<'a>(&'a self, destination: RenderTarget<'a>) -> RenderTarget<'a> {
        match self {
            Self::Direct(depth) => RenderTarget {
                depth_view: Some(depth),
                ..destination
            },
            Self::Hdr { target, .. } => RenderTarget {
                view: target.color_view(),
                resolve_target: target.resolve_target(),
                depth_view: Some(target.depth_view()),
                size: target.size(),
                config: target.render_target_config(),
            },
        }
    }

    pub fn finish(&self, encoder: &mut wgpu::CommandEncoder, destination: RenderTarget<'_>) {
        if let Self::Hdr { mapper, source, .. } = self {
            mapper.encode(encoder, source, destination);
        }
    }

    pub fn resize(
        &mut self,
        gpu: &GpuContext,
        size: PhysicalSize<u32>,
        final_config: RenderTargetConfig,
    ) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        match self {
            Self::Direct(depth) => {
                let (_, resized) = mikage::create_depth_texture(gpu, size, final_config);
                *depth = resized;
            }
            Self::Hdr {
                target,
                mapper,
                source,
            } => {
                target.resize(gpu, size).expect("resize HDR scene target");
                *source = mapper.source(gpu, target.view());
            }
        }
    }
}
