use std::sync::Arc;

use winit::window::Window as WinitWindow;

pub struct Window {
    winit_window: Arc<WinitWindow>,
    is_focused: bool,
    is_minimized: Option<bool>,
    width: u32,
    height: u32,
}

impl Window {
    pub fn new(winit_window: Arc<WinitWindow>) -> Self {
        let size = winit_window.inner_size();
        let is_minimized = winit_window.is_minimized();
        Window {
            winit_window,
            is_focused: false,
            is_minimized,
            width: size.width,
            height: size.height,
        }
    }

    pub fn get_winit_window(&self) -> Arc<WinitWindow> {
        self.winit_window.clone()
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

    pub fn get_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn set_title(&self, title: &str) {
        self.winit_window.set_title(title);
    }
}
