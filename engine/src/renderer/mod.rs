use crate::core::vertex::ElmVertex;
use crate::resource_manager::ResourceManager;
use anyhow::Result;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};

pub mod renderer_vulkan;

pub trait Renderer {
    fn new(resource_manager: &mut ResourceManager) -> Self
    where
        Self: std::marker::Sized;
    fn run(&mut self) -> Result<()>;
    fn on_update(&mut self) -> Result<()>;
    fn upload_mesh(&mut self, vertices: &[ElmVertex], indices: &[u32]) -> Result<()>;
    fn upload_texture(
        &mut self,
        image_data: &[u8],
        width: u32,
        height: u32,
        filter: (Option<MagFilter>, Option<MinFilter>),
        wrap: (WrappingMode, WrappingMode),
    ) -> Result<()>;
}
