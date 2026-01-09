use assets_manager::AssetCache;
use std::ops::Deref;

pub mod gltf_model;

pub struct AssetLoader {
    pub cache: AssetCache,
}

impl AssetLoader {
    pub fn new() -> Self {
        AssetLoader {
            cache: AssetCache::new("assets").expect("Failed to create asset cache"),
        }
    }
}

impl Deref for AssetLoader {
    type Target = AssetCache;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl Default for AssetLoader {
    fn default() -> Self {
        Self::new()
    }
}
