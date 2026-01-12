use crate::asset_loader::gltf_model::GltfModel;
use crate::renderer::renderer_vulkan::VulkanRenderer;
use crate::{
    asset_loader::AssetLoader, input::Input, logger::Logger, renderer::Renderer,
    resource_manager::ResourceManager, window::Window,
};
use std::sync::Arc;
use tracing::{debug, error};
use winit::event::WindowEvent;
use winit::window::Window as WinitWindow;

pub struct Application {
    resources: ResourceManager,
    _logger: Logger,
    renderer: Option<Box<dyn Renderer>>,
}

impl Application {
    pub fn new() -> Application {
        let _logger = Logger::new();
        let mut resources = ResourceManager::new();
        resources.add(Input::new());
        resources.add(AssetLoader::new());
        Application {
            resources,
            _logger,
            renderer: None,
        }
    }

    pub fn set_window(&mut self, window: Arc<WinitWindow>) {
        let app_window = Window::new(window);
        self.resources.add(app_window);
        let renderer = VulkanRenderer::new(&mut self.resources);
        self.renderer = Some(Box::new(renderer));
    }

    pub fn handle_window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Focused(is_focused) => {
                self.resources.get_mut::<Window>().set_focused(is_focused);
            }
            WindowEvent::Resized(new_size) => {
                self.resources
                    .get_mut::<Window>()
                    .set_size(new_size.width, new_size.height);
            }
            WindowEvent::KeyboardInput {
                device_id, event, ..
            } => {
                self.resources
                    .get_mut::<Input>()
                    .handle_keyboard_input(device_id, event);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.resources
                    .get_mut::<Input>()
                    .handle_mouse_input(state, button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.resources.get_mut::<Input>().handle_cursor(position);
            }
            _ => (),
        }
    }

    pub fn run(&mut self) {
        let renderer = self
            .renderer
            .as_mut()
            .expect("Renderer must be initialized before running the application");
        let asset_loader = self.resources.get_mut::<AssetLoader>();
        let handle = asset_loader.load::<GltfModel>("super_car.scene");
        if let Ok(handle) = handle {
            let model = handle.read();
            // info!("Model: {:?}", model);
            for mesh in model.meshes.iter() {
                for primitive in mesh.primitives.iter() {
                    if let Err(e) = renderer.upload_mesh(&primitive.vertices, &primitive.indices) {
                        error!("Failed to upload mesh: {:?}", e);
                    }
                }
            }
            for texture in model.textures.iter() {
                debug!("Texture: {:?}", texture);
                if let Some(index) = texture.image {
                    let image = &model.images[index];
                    if let Err(e) = renderer.upload_texture(
                        &image.pixels,
                        image.width,
                        image.height,
                        (texture.sampler.mag_filter, texture.sampler.min_filter),
                        (texture.sampler.wrap_s, texture.sampler.wrap_t),
                    ) {
                        error!("Failed to upload texture: {:?}", e);
                    }
                }
            }
        } else {
            error!("Failed to load super_car.scene: {:?}", handle);
        }

        if let Err(e) = renderer.run() {
            error!("Renderer encountered an error: {:?}", e);
        }
    }

    pub fn on_update(&mut self) {
        let renderer = self
            .renderer
            .as_mut()
            .expect("Renderer must be initialized before updating the application");
        let start_time = std::time::Instant::now();

        if let Err(e) = renderer.on_update() {
            error!("Renderer update error: {:?}", e);
            panic!("Renderer update failed");
        }

        {
            let window = self.resources.get_mut::<Window>();
            window.get_winit_window().request_redraw();
        }

        let input = self.resources.get_mut::<Input>();
        input.prepare_for_next_frame();

        let end_time = std::time::Instant::now();
        let frame_duration = end_time.duration_since(start_time);
        let ms = frame_duration.as_secs_f64() * 1000.0;
        let fps = if ms > 0.0 { 1000.0 / ms } else { 0.0 };

        // Update title with timing info
        {
            let window = self.resources.get_mut::<Window>();
            window.set_title(&format!("Elements | {:>5.2} ms | {:>5.1} FPS", ms, fps));
        }
    }
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}
