use std::sync::Arc;

use tracing::info;
use winit::event::WindowEvent;

use crate::{
    input::Input, logger::Logger, renderer::Renderer, resource_manager::ResourceManager,
    window::Window,
};

pub struct Application {
    resources: ResourceManager,
}

impl Application {
    pub fn new() -> Application {
        let mut resources = ResourceManager::new();
        resources.add(Input::new());
        let mut _logger = Logger::new();
        let mut _renderer = Renderer::new();
        Application { resources }
    }

    pub fn set_window(&mut self, window: Option<Arc<winit::window::Window>>) {
        let app_window = Window::new(window);
        self.resources.add(app_window);
    }

    pub fn handle_window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Focused(is_focused) => {
                self.resources.get_mut::<Window>().set_focued(is_focused);
            }
            WindowEvent::KeyboardInput {
                device_id, event, ..
            } => {
                let input = self.resources.get_mut::<Input>();
                input.handle_keyboard_input(device_id, event);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let input = self.resources.get_mut::<Input>();
                input.handle_mouse_input(state, button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let input = self.resources.get_mut::<Input>();
                input.handle_cursor(position);
            }
            WindowEvent::RedrawRequested => {
                let window = self.resources.get_mut::<Window>();
                window.get_window().unwrap().request_redraw();
                let input = self.resources.get_mut::<Input>();
                input.prepare_for_next_frame();
            }
            _ => (),
        }
    }
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}
