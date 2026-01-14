use tracing::Level;
use tracing_subscriber::FmtSubscriber;

pub struct Logger;

impl Logger {
    pub fn new() -> Self {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(Level::TRACE)
            .pretty()
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");

        Self
    }

    // fn on_update(&self) {}
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}
