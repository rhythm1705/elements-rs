use tracing::info;

use crate::{
    application::Application,
    platform::{Platform, platform_winit::WinitPlatform},
};

pub mod application;
pub mod input;
pub mod logger;
pub mod platform;
pub mod renderer;
pub mod resource_manager;
pub mod window;

fn main() {
    let app = Application::new();
    let platform = WinitPlatform::new(app);
    platform.run();
    info!("HELLO ELEMENTS!");
}
