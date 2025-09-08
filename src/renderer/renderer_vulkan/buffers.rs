use std::sync::Arc;

use anyhow::Result;
use glam::{Vec2, Vec3};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
    sync::GpuFuture,
};

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
pub struct MyVertex {
    // Every field needs to explicitly state the desired shader input format
    // The `name` attribute can be used to specify shader input names to match.
    // By default the field-name is used.
    #[name("inPosition")]
    #[format(R32G32_SFLOAT)]
    pub position: Vec2,

    #[name("inColor")]
    #[format(R32G32B32_SFLOAT)]
    pub color: Vec3,
}

pub struct RenderMesh {
    pub vertex_buffer: Subbuffer<[MyVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub vertex_count: u32,
    pub index_count: u32,
}

pub struct VulkanAllocator {
    memory_allocator: Arc<StandardMemoryAllocator>,
    meshes: Vec<RenderMesh>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl VulkanAllocator {
    pub fn new(
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device));
        Self {
            memory_allocator,
            meshes: Vec::new(),
            graphics_queue,
            command_buffer_allocator,
        }
    }

    pub fn create_mesh(&mut self, vertices: &mut [MyVertex], indices: &[u32]) -> Result<usize> {
        let vertex_buffer = self.create_vertex_buffer(vertices)?;
        let index_buffer = self.create_index_buffer(indices)?;

        let mesh = RenderMesh {
            vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            vertex_buffer,
            index_buffer,
        };
        self.meshes.push(mesh);
        Ok(self.meshes.len() - 1)
    }

    pub fn get_mesh(&self, mesh_id: usize) -> Option<&RenderMesh> {
        self.meshes.get(mesh_id)
    }

    fn create_vertex_buffer(&self, vertices: &mut [MyVertex]) -> Result<Subbuffer<[MyVertex]>> {
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.iter().cloned(),
        )?;

        let vertex_buffer = Buffer::new_slice::<MyVertex>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.len() as DeviceSize,
        )?;

        // Create a one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        cbb.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer,
            vertex_buffer.clone(),
        ))?;
        let cb = cbb.build()?;

        // Execute the copy command and wait for completion before proceeding.
        cb.execute(self.graphics_queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None /* timeout */)?;

        Ok(vertex_buffer)
    }

    fn create_index_buffer(&self, indices: &[u32]) -> Result<Subbuffer<[u32]>> {
        let staging_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.iter().cloned(),
        )?;

        let index_buffer = Buffer::new_slice::<u32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.len() as DeviceSize,
        )?;

        // Create a one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        cbb.copy_buffer(CopyBufferInfo::buffers(
            staging_buffer,
            index_buffer.clone(),
        ))?;
        let cb = cbb.build()?;

        // Execute the copy command and wait for completion before proceeding.
        cb.execute(self.graphics_queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None /* timeout */)?;

        Ok(index_buffer)
    }
}
