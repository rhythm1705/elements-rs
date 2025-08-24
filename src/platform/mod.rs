use crate::application::Application;

pub mod platform_winit;

/// Defines the contract for a platform layer.
/// Its only job is to take an application and run it.
pub trait Platform {
    /// Creates a new platform instance, wrapping the application.
    fn new(app: Application) -> Self;

    /// Runs the application, taking over the main thread.
    /// This function will never return.
    fn run(self);
}
