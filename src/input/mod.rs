use std::collections::HashSet;

use tracing::info;
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceId, ElementState, KeyEvent, MouseButton},
    keyboard::PhysicalKey,
};

#[derive(Default)]
pub struct Input {
    keys_pressed: HashSet<PhysicalKey>,
    keys_just_pressed: HashSet<PhysicalKey>,
    keys_just_released: HashSet<PhysicalKey>,
    mouse_buttons_pressed: HashSet<MouseButton>,
    mouse_pos: (f64, f64),
}

impl Input {
    pub fn new() -> Self {
        Input::default()
    }

    // This is crucial! Call this at the start of every frame to clear the "just" pressed/released states.
    pub fn prepare_for_next_frame(&mut self) {
        self.keys_just_pressed.clear();
        self.keys_just_released.clear();
    }

    // Helper functions to make querying input easy
    pub fn is_key_pressed(&self, key: PhysicalKey) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn was_key_just_pressed(&self, key: PhysicalKey) -> bool {
        self.keys_just_pressed.contains(&key)
    }

    pub fn handle_keyboard_input(&mut self, device_id: DeviceId, event: KeyEvent) {
        let keycode = event.physical_key;
        info!(
            "Keyboard input from {:?} for key {:?} with state {:?}",
            device_id, keycode, event.state
        );
        match event.state {
            ElementState::Pressed => {
                // If the key is not already being held down, it's "just pressed"
                if !self.keys_pressed.contains(&keycode) {
                    self.keys_just_pressed.insert(keycode);
                }
                self.keys_pressed.insert(keycode);
            }
            ElementState::Released => {
                self.keys_pressed.remove(&keycode);
                self.keys_just_released.insert(keycode);
            }
        }
    }

    pub fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        info!("Mouse pressed {:?}", button);
        match state {
            ElementState::Pressed => {
                self.mouse_buttons_pressed.insert(button);
            }
            ElementState::Released => {
                self.mouse_buttons_pressed.remove(&button);
            }
        }
    }

    pub fn handle_cursor(&mut self, position: PhysicalPosition<f64>) {
        info!("Mouse position {:?}", position);
        self.mouse_pos = (position.x, position.y);
    }
}
