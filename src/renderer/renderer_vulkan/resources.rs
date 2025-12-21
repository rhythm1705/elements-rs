use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use vulkano::image::sampler::SamplerAddressMode;

use anyhow::{Result, anyhow};
use glam::{Mat4, Vec2, Vec3};
use image::ImageReader;
use vulkano::command_buffer::{CopyBufferToImageInfo, PrimaryAutoCommandBuffer};
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::sampler::BorderColor::IntOpaqueBlack;
use vulkano::image::sampler::SamplerMipmapMode::Linear;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageTiling, ImageType, ImageUsage};
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
    // By default, the field-name is used.
    #[name("inPosition")]
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,

    #[name("inColor")]
    #[format(R32G32B32_SFLOAT)]
    pub color: Vec3,

    #[name("inTexCoord")]
    #[format(R32G32_SFLOAT)]
    pub tex_coord: Vec2,
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

pub struct RenderMesh {
    pub vertex_buffer: Subbuffer<[MyVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub _vertex_count: u32,
    pub index_count: u32,
}

#[allow(dead_code)]
pub struct Texture {
    pub image: Arc<Image>,
    pub image_view: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
}

pub struct VulkanResources {
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    meshes: Vec<RenderMesh>,
    depth_resource: Option<Arc<ImageView>>,
    textures: HashMap<String, Texture>,
    uniform_buffers: Vec<Subbuffer<UniformBufferObject>>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl VulkanResources {
    pub fn new(
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        Self {
            device,
            memory_allocator,
            meshes: Vec::new(),
            depth_resource: None,
            textures: HashMap::new(),
            uniform_buffers: Vec::new(),
            graphics_queue,
            command_buffer_allocator,
        }
    }

    pub fn create_mesh(&mut self, vertices: &mut [MyVertex], indices: &[u32]) -> Result<usize> {
        let vertex_buffer = self.create_vertex_buffer(vertices)?;
        let index_buffer = self.create_index_buffer(indices)?;

        let mesh = RenderMesh {
            _vertex_count: vertices.len() as u32,
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

    pub fn create_texture(&mut self, path: &Path) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        if self.textures.contains_key(&path_str) {
            return Ok(());
        }

        let image = self.create_texture_image(path)?;
        let image_view = self.create_texture_image_view(image.clone())?;
        let sampler = self.create_texture_sampler()?;

        let texture = Texture {
            image,
            image_view,
            sampler,
        };
        self.textures.insert(path_str, texture);
        Ok(())
    }

    pub fn get_texture(&self, path: &Path) -> Option<&Texture> {
        let path_str = path.to_string_lossy().to_string();
        self.textures.get(&path_str)
    }

    fn create_texture_image(&self, path: &Path) -> Result<Arc<Image>> {
        let img = ImageReader::open(path)?.decode()?.to_rgba8();
        let (width, height) = img.dimensions();
        let staging_buffer = self.create_staging_buffer(&img.into_raw())?;

        let texture_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        self.copy_buffer_to_image(staging_buffer, texture_image.clone())?;

        Ok(texture_image)
    }

    fn create_texture_image_view(&self, image: Arc<Image>) -> Result<Arc<ImageView>> {
        let image_view = ImageView::new_default(image)?;
        Ok(image_view)
    }

    fn create_texture_sampler(&self) -> Result<Arc<Sampler>> {
        let sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                anisotropy: Some(
                    self.device
                        .physical_device()
                        .properties()
                        .max_sampler_anisotropy,
                ),
                border_color: IntOpaqueBlack,
                mipmap_mode: Linear,
                ..Default::default()
            },
        )?;
        Ok(sampler)
    }

    pub fn create_depth_resources(&mut self, extent: [u32; 2]) -> Result<()> {
        let depth_format = self.find_depth_format()?;

        let depth_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: depth_format,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let depth_image_view = ImageView::new_default(depth_image.clone())?;
        self.depth_resource = Some(depth_image_view);
        Ok(())
    }

    pub fn get_depth_resources(&self) -> Result<Arc<ImageView>> {
        self.depth_resource
            .as_ref()
            .cloned()
            .ok_or(anyhow!("Depth resources not created"))
    }

    fn find_supported_format(
        &self,
        candidates: &[Format],
        tiling: ImageTiling,
        features: FormatFeatures,
    ) -> Result<Format> {
        for &format in candidates {
            let properties = self.device.physical_device().format_properties(format);
            let supported = match tiling {
                ImageTiling::Linear => properties?.linear_tiling_features.intersects(features),
                ImageTiling::Optimal => properties?.optimal_tiling_features.intersects(features),
                _ => false,
            };
            if supported {
                return Ok(format);
            }
        }
        Err(anyhow!("No supported format found"))
    }

    pub fn find_depth_format(&self) -> Result<Format> {
        self.find_supported_format(
            &[
                Format::D32_SFLOAT,
                Format::D32_SFLOAT_S8_UINT,
                Format::D24_UNORM_S8_UINT,
            ],
            ImageTiling::Optimal,
            FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn _has_stencil_component(&self, format: Format) -> bool {
        matches!(
            format,
            Format::D32_SFLOAT_S8_UINT | Format::D24_UNORM_S8_UINT
        )
    }

    fn copy_buffer_to_image<T: BufferContents + Clone>(
        &self,
        src_buffer: Subbuffer<[T]>,
        dst_image: Arc<Image>,
    ) -> Result<()> {
        let mut cbb = self.begin_single_time_commands()?;
        cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src_buffer, dst_image))?;
        self.end_single_time_commands(cbb)?;
        Ok(())
    }

    fn create_vertex_buffer(&self, vertices: &[MyVertex]) -> Result<Subbuffer<[MyVertex]>> {
        let staging_buffer = self.create_staging_buffer(vertices)?;

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

        self.copy_buffer(staging_buffer, vertex_buffer.clone())?;

        Ok(vertex_buffer)
    }

    fn create_index_buffer(&self, indices: &[u32]) -> Result<Subbuffer<[u32]>> {
        let staging_buffer = self.create_staging_buffer(indices)?;

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

        self.copy_buffer(staging_buffer, index_buffer.clone())?;

        Ok(index_buffer)
    }

    fn create_staging_buffer<T: BufferContents + Clone>(
        &self,
        data: &[T],
    ) -> Result<Subbuffer<[T]>> {
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
            data.iter().cloned(),
        )?;
        Ok(staging_buffer)
    }

    fn copy_buffer<T: BufferContents + Clone>(
        &self,
        src: Subbuffer<[T]>,
        dst: Subbuffer<[T]>,
    ) -> Result<()> {
        let mut cbb = self.begin_single_time_commands()?;
        cbb.copy_buffer(CopyBufferInfo::buffers(src, dst))?;
        self.end_single_time_commands(cbb)?;
        Ok(())
    }

    fn begin_single_time_commands(
        &self,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        Ok(command_buffer)
    }

    fn end_single_time_commands(
        &self,
        command_buffer: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        let cb = command_buffer.build()?;
        cb.execute(self.graphics_queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None /* timeout */)?;
        Ok(())
    }

    pub fn create_uniform_buffers(&mut self, count: usize) -> Result<()> {
        if self.uniform_buffers.len() == count {
            return Ok(()); // Already sized correctly.
        }
        self.uniform_buffers.clear();
        for _ in 0..count {
            let uniform_buffer = Buffer::new_sized::<UniformBufferObject>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )?;
            self.uniform_buffers.push(uniform_buffer);
        }
        Ok(())
    }

    pub fn get_uniform_buffer(&self, index: usize) -> Option<Subbuffer<UniformBufferObject>> {
        self.uniform_buffers.get(index).cloned()
    }
}
