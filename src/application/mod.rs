use std::sync::Arc;

use crate::{
    input::Input, logger::Logger, renderer::Renderer, resource_manager::ResourceManager,
    window::Window,
};
use winit::event::WindowEvent;
use winit::window::Window as WinitWindow;

pub struct Application {
    resources: ResourceManager,
    _logger: Logger,
    renderer: Renderer,
}

impl Application {
    pub fn new() -> Application {
        let _logger = Logger::new();
        let mut resources = ResourceManager::new();
        resources.add(Input::new());
        let renderer = Renderer::new();
        Application {
            resources,
            _logger,
            renderer,
        }
    }

    pub fn set_window(&mut self, window: Arc<WinitWindow>) {
        let app_window = Window::new(window);
        self.resources.add(app_window);
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
        self.renderer.run(&mut self.resources);
    }

    pub fn on_update(&mut self) {
        let start_time = std::time::Instant::now();

        self.renderer.on_update();

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
