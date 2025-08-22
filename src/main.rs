use tracing::info;

use crate::{logging_system::LoggingManager, window_manager::WindowManager};

pub mod logging_system;
pub mod window_manager;

fn main() {
    LoggingManager::new();
    WindowManager::new();
    info!("HELLO ELEMENTS!");
}
