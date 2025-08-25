use std::sync::Arc;

use crate::{
    input::Input, logger::Logger, renderer::Renderer, resource_manager::ResourceManager,
    window::Window,
};
use winit::event::WindowEvent;
use winit::window::Window as WinitWindow;

pub struct Application {
    resources: ResourceManager,
    logger: Logger,
    renderer: Renderer,
}

impl Application {
    pub fn new() -> Application {
        let mut logger = Logger::new();
        let mut resources = ResourceManager::new();
        resources.add(Input::new());
        let renderer = Renderer::new();
        Application {
            resources,
            logger,
            renderer,
        }
    }

    pub fn set_window(&mut self, window: Option<Arc<WinitWindow>>) {
        let app_window = Window::new(window);
        self.resources.add(app_window);
    }

    pub fn handle_window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Focused(is_focused) => {
                self.resources.get_mut::<Window>().set_focused(is_focused);
            }
            WindowEvent::Resized(new_size) => {
                let window = self.resources.get_mut::<Window>();
                window.set_size(new_size.width, new_size.height);
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
            _ => (),
        }
    }

    pub fn run(&mut self) {
        self.renderer.run(&self.resources);
    }

    pub fn on_update(&mut self) {
        self.renderer.on_update(&mut self.resources);
        let window = self.resources.get_mut::<Window>();
        window.get_window().unwrap().request_redraw();
        let input = self.resources.get_mut::<Input>();
        input.prepare_for_next_frame();
    }
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}
