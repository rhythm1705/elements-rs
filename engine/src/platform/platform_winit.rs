use std::sync::Arc;

use crate::engine::Engine;
use crate::platform::Platform;
use tracing::info;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window as WinitWindow, WindowId};

pub struct WinitPlatform {
    app: Engine,
}

impl ApplicationHandler for WinitPlatform {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let winit_window = Arc::new(
            event_loop
                .create_window(WinitWindow::default_attributes())
                .unwrap(),
        );
        info!("Created window with ID: {:?}", winit_window.id());
        winit_window.set_title("Elements");
        self.app.set_window(winit_window);
        self.app.run();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // TODO: Create platform agnostic window events
        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.app.on_update();
            }
            other => self.app.handle_window_event(other),
        }
    }
}

impl Platform for WinitPlatform {
    fn new(app: Engine) -> Self {
        Self { app }
    }

    fn run(&mut self) {
        info!("Starting event loop...");

        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);
        let _ = event_loop.run_app(self);
    }
}
