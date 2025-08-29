use std::{error::Error, sync::Arc};

use tracing::error;
use vulkano::{
    Validated, VulkanError,
    device::Device,
    format::Format,
    image::{Image, ImageUsage},
    swapchain::{
        ColorSpace, PresentMode, Surface, SurfaceCapabilities, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, acquire_next_image,
    },
};
use winit::window::Window as WinitWindow;

struct SwapchainSupportDetails {
    capabilities: SurfaceCapabilities,
    formats: Vec<Format>,
    present_modes: Vec<PresentMode>,
}

struct VulkanSwapchain {
    swapchain: Arc<Swapchain>,
    // support_details: SwapchainSupportDetails,
    images: Vec<Arc<Image>>,
    format: Format,
    extent: [u32; 2],
}

impl VulkanSwapchain {
    pub fn new(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window: &WinitWindow,
    ) -> Result<Self, Box<dyn Error>> {
        let window_size = window.inner_size();
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
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
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
                        .ok_or("No supported composite alpha")?,
                    ..Default::default()
                },
            )?
        };

        let format = swapchain.image_format();
        let extent = swapchain.image_extent();

        Ok(VulkanSwapchain {
            swapchain,
            images,
            format,
            extent,
        })
    }

    pub fn recreate(&mut self, window_size: [u32; 2]) -> Result<(), Box<dyn Error>> {
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
    ) -> Result<(u32, bool, SwapchainAcquireFuture), Box<dyn Error>> {
        Ok(acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap)?)
    }

    pub fn present(&self) {}
}
