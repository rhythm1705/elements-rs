use glam::{Vec2, Vec3};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[repr(C)]
#[derive(BufferContents, PartialEq, Debug, Clone, Copy)]
pub struct ElmVec3(Vec3);

impl Deref for ElmVec3 {
    type Target = glam::Vec3;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<glam::Vec3> for ElmVec3 {
    fn from(v: glam::Vec3) -> Self {
        Self(v)
    }
}

impl Eq for ElmVec3 {}

impl Hash for ElmVec3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for f in &self.to_array() {
            f.to_bits().hash(state);
        }
    }
}

#[repr(C)]
#[derive(BufferContents, PartialEq, Debug, Clone, Copy)]
pub struct ElmVec2(Vec2);

impl Deref for ElmVec2 {
    type Target = glam::Vec2;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Vec2> for ElmVec2 {
    fn from(v: Vec2) -> Self {
        Self(v)
    }
}

impl Eq for ElmVec2 {}
impl Hash for ElmVec2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for f in &self.to_array() {
            f.to_bits().hash(state);
        }
    }
}

#[repr(C)]
#[derive(BufferContents, Vertex, Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ElmVertex {
    // Every field needs to explicitly state the desired shader input format
    // The `name` attribute can be used to specify shader input names to match.
    // By default, the field-name is used.
    #[name("inPosition")]
    #[format(R32G32B32_SFLOAT)]
    pub position: ElmVec3,

    #[name("inColor")]
    #[format(R32G32B32_SFLOAT)]
    pub color: ElmVec3,

    #[name("inTexCoord")]
    #[format(R32G32_SFLOAT)]
    pub tex_coord: ElmVec2,
}
