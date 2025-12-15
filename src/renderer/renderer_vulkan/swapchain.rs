use std::sync::Arc;

use anyhow::{anyhow, Result};
use vulkano::{
    device::Device, format::Format,
    image::{Image, ImageUsage},
    swapchain::{
        acquire_next_image, ColorSpace, Surface, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo,
    },
    Validated,
    VulkanError,
};

use crate::renderer::renderer_vulkan::MAX_FRAMES_IN_FLIGHT;

// TODO: Implement querying swapchain support details
// struct SwapchainSupportDetails {
//     capabilities: SurfaceCapabilities,
//     formats: Vec<Format>,
//     present_modes: Vec<PresentMode>,
// }

pub struct VulkanSwapchain {
    pub swapchain: Arc<Swapchain>,
    // support_details: SwapchainSupportDetails,
    // pub surface: Arc<Surface>,
    pub images: Vec<Arc<Image>>,
    pub format: Format,
    pub extent: [u32; 2],
}

impl VulkanSwapchain {
    pub fn new(device: Arc<Device>, surface: Arc<Surface>, window_size: [u32; 2]) -> Result<Self> {
        let (swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())?;

            // Choosing the internal format that the images will have.
            let (image_format, _) = {
                let formats = device
                    .physical_device()
                    .surface_formats(&surface, Default::default())?;
                // Prefer sRGB non-linear formats
                formats
                    .iter()
                    .find(|(f, c)| {
                        f.ycbcr_chroma_sampling().is_none()
                            && *f == Format::R8G8B8A8_SRGB
                            && *c == ColorSpace::SrgbNonLinear
                    })
                    .cloned()
                    .unwrap_or_else(|| formats[0])
            };

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities
                        .min_image_count
                        .max(MAX_FRAMES_IN_FLIGHT as u32),
                    image_format,
                    image_extent: window_size,
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    present_mode: surface_capabilities
                        .compatible_present_modes
                        .iter()
                        .find(|m| **m == vulkano::swapchain::PresentMode::Mailbox)
                        .copied()
                        .unwrap_or(vulkano::swapchain::PresentMode::Fifo),
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .ok_or(anyhow!("No supported composite alpha"))?,
                    ..Default::default()
                },
            )?
        };

        let format = swapchain.image_format();
        let extent = swapchain.image_extent();

        Ok(VulkanSwapchain {
            swapchain,
            // surface,
            images,
            format,
            extent,
        })
    }

    pub fn recreate(&mut self, window_size: [u32; 2]) -> Result<()> {
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window_size,
            ..self.swapchain.create_info()
        })?;
        self.swapchain = new_swapchain;
        self.images = new_images;
        self.extent = window_size;
        Ok(())
    }

    pub fn acquire_next_image(
        &self,
    ) -> Result<(u32, bool, SwapchainAcquireFuture), Validated<VulkanError>> {
        Ok(acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap)?)
    }

    // TODO: Implement present function
    // pub fn present(&self) {}
}
