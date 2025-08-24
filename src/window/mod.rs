use std::sync::Arc;

use winit::window::Window as WinitWindow;

#[derive(Default)]
pub struct Window {
    window: Option<Arc<WinitWindow>>,
    is_focused: bool,
}

impl Window {
    pub fn new(winit_window: Option<Arc<WinitWindow>>) -> Self {
        Window {
            window: winit_window,
            is_focused: false,
        }
    }

    pub fn get_window(&mut self) -> Option<Arc<WinitWindow>> {
        self.window.clone()
    }

    pub fn set_focued(&mut self, focused: bool) {
        self.is_focused = focused;
    }

    pub fn is_focused(self) -> bool {
        self.is_focused
    }
}
