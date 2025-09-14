use crate::{
    renderer::renderer_vulkan::VulkanRenderer, resource_manager::ResourceManager, window::Window,
};

pub mod renderer_vulkan;

pub struct Renderer {
    vk_renderer: Option<VulkanRenderer>,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer { vk_renderer: None }
    }

    pub fn run(&mut self, resources: &ResourceManager) {
        match VulkanRenderer::new(resources) {
            Ok(vk) => {
                self.vk_renderer = Some(vk);
            }
            Err(e) => {
                panic!("Could not initialize vulkan renderer: {:?}", e);
            }
        }

        let init_result = if let Some(vk) = self.vk_renderer.as_mut() {
            vk.initialize_render_context()
        } else {
            panic!("Vulkan renderer not available; skipping render context initialization");
        };

        if let Err(e) = init_result {
            panic!("Failed to initialize render context: {:?}", e);
        }
    }

    pub fn on_update(&mut self, resources: &mut ResourceManager) {
        let window = resources.get::<Window>();
        if let Some(vk) = &mut self.vk_renderer
            && let Err(e) = vk.draw_frame(window)
        {
            panic!("Error during draw frame: {:?}", e);
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}
