pub(crate) use crate::core::ubo::UniformBufferObject;
pub(crate) use crate::core::vertex::ElmVertex;
use anyhow::{Result, anyhow};
use std::cmp::max;
use std::sync::Arc;
use vulkano::command_buffer::{
    BlitImageInfo, CopyBufferToImageInfo, ImageBlit, PrimaryAutoCommandBuffer,
};
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::sampler::BorderColor::IntOpaqueBlack;
use vulkano::image::sampler::SamplerMipmapMode::Linear;
use vulkano::image::sampler::{Filter, LOD_CLAMP_NONE, SamplerAddressMode};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{
    Image, ImageAspect, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageTiling,
    ImageType, ImageUsage, SampleCount,
};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

pub struct GPUMesh {
    pub vertex_buffer: Subbuffer<[ElmVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub _vertex_count: u32,
    pub index_count: u32,
}

pub struct GPUTexture {
    pub image_view: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
}

pub struct VulkanResources {
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub msaa_samples: SampleCount,
    pub meshes: Vec<GPUMesh>,
    pub textures: Vec<GPUTexture>,
    pub depth_resource: Option<Arc<ImageView>>,
    pub color_resource: Option<Arc<ImageView>>,
    pub uniform_buffers: Vec<Subbuffer<UniformBufferObject>>,
}

impl VulkanResources {
    pub fn new(
        device: Arc<Device>,
        graphics_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let properties = device.physical_device().properties();
        let msaa_samples = properties
            .framebuffer_color_sample_counts
            .union(properties.framebuffer_depth_sample_counts)
            .max_count();
        Self {
            device,
            graphics_queue,
            memory_allocator,
            command_buffer_allocator,
            msaa_samples,
            meshes: Vec::new(),
            textures: Vec::new(),
            depth_resource: None,
            color_resource: None,
            uniform_buffers: Vec::new(),
        }
    }

    pub fn upload_mesh(&mut self, vertices: &[ElmVertex], indices: &[u32]) -> Result<()> {
        let vertex_buffer = self.create_vertex_buffer(vertices)?;
        let index_buffer = self.create_index_buffer(indices)?;

        let mesh = GPUMesh {
            _vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            vertex_buffer,
            index_buffer,
        };
        self.meshes.push(mesh);
        Ok(())
    }

    pub fn get_mesh(&self, mesh_id: usize) -> Option<&GPUMesh> {
        self.meshes.get(mesh_id)
    }

    pub fn upload_texture(
        &mut self,
        image_data: &[u8],
        width: u32,
        height: u32,
        mag_filter: Filter,
        min_filter: Filter,
        address_mode: [SamplerAddressMode; 3],
    ) -> Result<()> {
        let mip_levels = max(width, height).ilog2() + 1;
        let image = self.create_texture_image(image_data, width, height, mip_levels)?;
        let image_view = ImageView::new_default(image.clone())?;
        let sampler = self.create_texture_sampler(image, mag_filter, min_filter, address_mode)?;

        let texture = GPUTexture {
            image_view,
            sampler,
        };
        self.textures.push(texture);
        Ok(())
    }

    pub fn get_texture(&self, texture_id: usize) -> Option<&GPUTexture> {
        self.textures.get(texture_id)
    }

    fn create_texture_image(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<Arc<Image>> {
        let staging_buffer = self.create_staging_buffer(image_data)?;

        let texture_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent: [width, height, 1],
                mip_levels,
                array_layers: 1,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        self.copy_buffer_to_image(staging_buffer, texture_image.clone())?;

        self.generate_mipmaps(texture_image.clone())?;

        Ok(texture_image)
    }

    fn generate_mipmaps(&self, image: Arc<Image>) -> Result<()> {
        let format_properties = self
            .device
            .physical_device()
            .format_properties(image.format())?;

        if !format_properties
            .optimal_tiling_features
            .intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return Err(anyhow!(
                "Texture image format does not support linear blitting!"
            ));
        }

        let mut command_buffer = self.begin_single_time_commands()?;

        let mut mip_width = image.extent()[0];
        let mut mip_height = image.extent()[1];

        // NOTE: This function assumes that mip level 0 has already been populated
        // (typically via a preceding copy_buffer_to_image call) and is in a layout
        // that allows it to be used as a transfer source for linear blits.
        debug_assert!(
            image.mip_levels() > 0,
            "Image must have at least one mip level"
        );
        for level in 1..image.mip_levels() {
            let next_mip_width = if mip_width > 1 { mip_width / 2 } else { 1 };
            let next_mip_height = if mip_height > 1 { mip_height / 2 } else { 1 };

            command_buffer.blit_image(BlitImageInfo {
                src_image: image.clone(),
                src_image_layout: ImageLayout::TransferSrcOptimal,
                dst_image: image.clone(),
                dst_image_layout: ImageLayout::TransferDstOptimal,
                regions: vec![ImageBlit {
                    src_subresource: ImageSubresourceLayers {
                        aspects: ImageAspect::Color.into(),
                        mip_level: level - 1,
                        array_layers: 0..1,
                    },
                    src_offsets: [[0, 0, 0], [mip_width, mip_height, 1]],
                    dst_subresource: ImageSubresourceLayers {
                        aspects: ImageAspect::Color.into(),
                        mip_level: level,
                        array_layers: 0..1,
                    },
                    dst_offsets: [[0, 0, 0], [next_mip_width, next_mip_height, 1]],
                    ..ImageBlit::default()
                }]
                .into(),
                filter: Filter::Linear,
                ..BlitImageInfo::images(image.clone(), image.clone())
            })?;

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        self.end_single_time_commands(command_buffer)?;

        Ok(())
    }

    fn create_texture_sampler(
        &self,
        _image: Arc<Image>,
        mag_filter: Filter,
        min_filter: Filter,
        address_mode: [SamplerAddressMode; 3],
    ) -> Result<Arc<Sampler>> {
        let sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter,
                min_filter,
                address_mode,
                anisotropy: Some(
                    self.device
                        .physical_device()
                        .properties()
                        .max_sampler_anisotropy,
                ),
                border_color: IntOpaqueBlack,
                mipmap_mode: Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )?;
        Ok(sampler)
    }

    pub fn msaa_samples(&self) -> SampleCount {
        self.msaa_samples
    }

    pub fn create_color_resources(&mut self, extent: [u32; 2], format: Format) -> Result<()> {
        let color_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                samples: self.msaa_samples,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;

        let color_image_view = ImageView::new_default(color_image.clone())?;
        self.color_resource = Some(color_image_view);
        Ok(())
    }

    pub fn get_color_resources(&self) -> Result<Arc<ImageView>> {
        self.color_resource
            .as_ref()
            .cloned()
            .ok_or(anyhow!("Color resources not created"))
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
                samples: self.msaa_samples,
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

    fn create_vertex_buffer(&self, vertices: &[ElmVertex]) -> Result<Subbuffer<[ElmVertex]>> {
        let staging_buffer = self.create_staging_buffer(vertices)?;

        let vertex_buffer = Buffer::new_slice::<ElmVertex>(
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
