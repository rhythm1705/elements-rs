use glam::Mat4;
use vulkano::buffer::BufferContents;

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}
