use std::sync::Arc;

use winit::window::Window as WinitWindow;

pub struct Window {
    window: Option<Arc<WinitWindow>>,
    is_focused: bool,
    is_minimized: Option<bool>,
    width: u32,
    height: u32,
}

impl Window {
    pub fn new(winit_window: Option<Arc<WinitWindow>>) -> Self {
        let size;
        let is_minimized;
        if let Some(w) = &winit_window {
            size = w.inner_size();
            is_minimized = w.is_minimized();
        } else {
            size = winit::dpi::PhysicalSize::new(0, 0);
            is_minimized = Some(false);
        };
        Window {
            window: winit_window,
            is_focused: false,
            is_minimized,
            width: size.width,
            height: size.height,
        }
    }

    pub fn get_window(&self) -> Option<Arc<WinitWindow>> {
        self.window.clone()
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.is_focused = focused;
    }

    pub fn is_focused(self) -> bool {
        self.is_focused
    }

    pub fn is_minimized(&self) -> bool {
        self.is_minimized.unwrap_or(false)
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }
}
