use crate::engine::Engine;
use crate::platform::Platform;
use crate::platform::platform_winit::WinitPlatform;

pub struct Application {
    platform: Box<dyn Platform>,
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

impl Application {
    pub fn new() -> Application {
        let engine = Engine::new();
        let platform = WinitPlatform::new(engine);
        Application {
            platform: Box::new(platform),
        }
    }

    pub fn run(&mut self) {
        self.platform.run();
    }
}
