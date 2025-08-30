use std::{
    collections::HashMap,
    sync::Arc,
};
use vulkano::{
    image::{Image, view::ImageView},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

pub struct RenderTargets {
    images: Vec<Arc<Image>>,
    // Cache: one framebuffer per image for each render pass
    framebuffers: HashMap<usize, Vec<Arc<Framebuffer>>>,
}

impl RenderTargets {
    pub fn new(images: Vec<Arc<Image>>) -> Self {
        Self {
            images,
            framebuffers: HashMap::new(),
        }
    }

    pub fn rebuild_for_pass(
        &mut self,
        pass_key: usize,
        render_pass: &Arc<RenderPass>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fbs = self
            .images
            .iter()
            .map(|img| {
                let view = ImageView::new_default(img.clone())?;
                Ok::<Arc<Framebuffer>, Box<dyn std::error::Error>>(Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.framebuffers.insert(pass_key, fbs);
        Ok(())
    }

    pub fn framebuffers(&self, pass_key: usize) -> Option<&[Arc<Framebuffer>]> {
        self.framebuffers.get(&pass_key).map(|v| v.as_slice())
    }

    pub fn replace_images(&mut self, images: Vec<Arc<Image>>) {
        self.images = images;
        self.framebuffers.clear(); // invalidate caches
    }
}
