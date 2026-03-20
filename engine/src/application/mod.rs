use crate::engine::Engine;
use crate::platform::Platform;
use crate::platform::platform_winit::WinitPlatform;
use crate::resource_manager::ResourceManager;
use hecs::World;

pub struct Application {
    engine: Engine,
}

impl Default for Application {
    fn default() -> Self {
        Self::new()
    }
}

impl Application {
    pub fn new() -> Application {
        Application {
            engine: Engine::new(),
        }
    }

    pub fn add_startup_system<F>(&mut self, system: F)
    where
        F: FnMut(&mut World, &mut ResourceManager) + 'static,
    {
        self.engine.add_startup_system(system);
    }

    pub fn add_update_system<F>(&mut self, system: F)
    where
        F: FnMut(&mut World, &mut ResourceManager) + 'static,
    {
        self.engine.add_update_system(system);
    }

    pub fn run(self) {
        let platform: Box<dyn Platform> = Box::new(WinitPlatform::new(self.engine));
        platform.run();
    }
}
