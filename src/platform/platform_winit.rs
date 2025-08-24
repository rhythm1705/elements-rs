use std::sync::Arc;

use crate::application::Application;
use crate::platform::Platform;
use crate::window::Window;
use tracing::info;
use vulkano::sync::event;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window as WinitWindow, WindowId};

use crate::input::Input;

pub struct WinitPlatform {
    app: Application,
}

impl ApplicationHandler for WinitPlatform {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let winit_window = Some(Arc::new(
            event_loop
                .create_window(WinitWindow::default_attributes())
                .unwrap(),
        ));
        self.app.set_window(winit_window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // TODO: Create platform agnostic window events
        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            other => self.app.handle_window_event(other),
        }
    }
}

// Here is the new part: We implement our abstract Platform trait.
impl Platform for WinitPlatform {
    fn new(app: Application) -> Self {
        Self { app }
    }

    fn run(mut self) {
        println!("[WinitPlatform] Starting event loop...");

        // EventLoop creation is now hidden inside the platform's run method.
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        // This is the magic. We tell winit's event loop to run using
        // `self` (the WinitPlatform instance) as the application handler.
        let _ = event_loop.run_app(&mut self);
    }
}
