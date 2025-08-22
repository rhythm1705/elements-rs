use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub struct LoggingManager {}

impl LoggingManager {
    pub fn new() -> Self {
        let subscriber = FmtSubscriber::builder()
            // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
            // will be written to stdout.
            .with_max_level(Level::TRACE)
            .pretty()
            // completes the builder.
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");

        Self {}
    }

    // fn on_update(&self) {}
}

impl Default for LoggingManager {
    fn default() -> Self {
        Self::new()
    }
}
