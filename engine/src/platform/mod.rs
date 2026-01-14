use crate::engine::Engine;

pub mod platform_winit;

/// Defines the contract for a platform layer.
/// Its only job is to take an engine and run it.
pub trait Platform {
    /// Creates a new platform instance, wrapping the engine.
    fn new(engine: Engine) -> Self
    where
        Self: Sized;

    /// Runs the engine, taking over the main thread.
    /// This function will never return.
    fn run(&mut self);
}
