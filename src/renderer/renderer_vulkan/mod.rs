use crate::renderer::Renderer;
use crate::renderer::renderer_vulkan::render_context::FrameState;
pub(crate) use crate::{
    renderer::renderer_vulkan::{
        pipeline::VulkanPipeline,
        render_context::{ActiveFrame, RenderContext},
        resources::{ElmVertex, VulkanResources},
        swapchain::VulkanSwapchain,
    },
    resource_manager::ResourceManager,
    window::Window,
};
use anyhow::{Context, Result, anyhow};
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use std::time::Duration;
use std::{sync::Arc, thread, time::Instant};
#[cfg(debug_assertions)]
use tracing::debug;
use tracing::{Level, info, span};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::DeviceFeatures;
use vulkano::image::sampler::{Filter, SamplerAddressMode};
#[cfg(debug_assertions)]
use vulkano::instance::debug::{
    DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
    DebugUtilsMessengerCreateInfo,
};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::window::Window as WinitWindow;

mod pipeline;
mod render_context;
pub mod resources;
mod shaders;
mod swapchain;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    winit_window: Arc<WinitWindow>,
    instance: Arc<Instance>,
    #[cfg(debug_assertions)]
    _debug_callback: DebugUtilsMessenger,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub resources: VulkanResources,
    render_context: Option<RenderContext>,
}

impl Renderer for VulkanRenderer {
    fn new(resource_manager: &mut ResourceManager) -> Self {
        let winit_window = resource_manager.get::<Window>().get_winit_window();

        let vk_lib = VulkanLibrary::new().unwrap();

        let enable_validation = cfg!(debug_assertions);

        let layers = if enable_validation {
            vec!["VK_LAYER_KHRONOS_validation".to_owned()]
        } else {
            Vec::new()
        };
        let mut required_extensions = Surface::required_extensions(&winit_window).unwrap();
        if enable_validation {
            required_extensions.ext_debug_utils = true;
            info!("Vulkan validation layers enabled");
        }

        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                enabled_layers: layers,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        #[cfg(debug_assertions)]
        let _debug_callback = DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                    match message_severity {
                        DebugUtilsMessageSeverity::ERROR => {
                            debug!(
                                "Vulkan Debug - ERROR - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::WARNING => {
                            debug!(
                                "Vulkan Debug - WARNING - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::INFO => {
                            debug!(
                                "Vulkan Debug - INFO - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::VERBOSE => {
                            debug!(
                                "Vulkan Debug - VERBOSE - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        _ => {
                            debug!(
                                "Vulkan Debug - UNKNOWN - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                    }
                })
            }),
        )
        .with_context(|| "Failed to create debug callback")
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                info!(
                    "Found device: {} (type: {:?})",
                    p.properties().device_name,
                    p.properties().device_type
                );
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, &winit_window)
                                .unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .with_context(|| "No suitable physical device found")
            .unwrap();

        info!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues_iter) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    sampler_anisotropy: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let graphics_queue: Arc<Queue> = queues_iter
            .next()
            .with_context(|| "No queue found")
            .unwrap();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let resources = VulkanResources::new(
            device.clone(),
            graphics_queue.clone(),
            command_buffer_allocator.clone(),
        );

        VulkanRenderer {
            winit_window,
            instance,
            #[cfg(debug_assertions)]
            _debug_callback,
            device,
            graphics_queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            resources,
            render_context: None,
        }
    }

    fn run(&mut self) -> Result<()> {
        let surface = Surface::from_window(self.instance.clone(), self.winit_window.clone())?;
        let window_size = self.winit_window.inner_size();

        let swapchain =
            VulkanSwapchain::new(self.device.clone(), surface.clone(), window_size.into())?;

        self.resources.create_depth_resources(swapchain.extent)?;

        let pipeline = VulkanPipeline::new(
            self.device.clone(),
            swapchain.format,
            self.resources.find_depth_format()?,
        )?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        self.resources
            .create_uniform_buffers(MAX_FRAMES_IN_FLIGHT)?;

        let descriptor_set = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let mut descriptor_writes = vec![];

                let ubo = self
                    .resources
                    .get_uniform_buffer(i)
                    .with_context(|| format!("Uniform buffer {i} not found"))?;
                descriptor_writes.push(WriteDescriptorSet::buffer(0, ubo));

                for texture in self.resources.textures.iter() {
                    descriptor_writes.push(WriteDescriptorSet::image_view_sampler(
                        descriptor_writes.len() as u32,
                        texture.image_view.clone(),
                        texture.sampler.clone(),
                    ));
                }

                let set = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    pipeline.layout().set_layouts()[0].clone(),
                    descriptor_writes,
                    [],
                )?;
                Ok(set)
            })
            .collect::<Result<Vec<_>>>()?;

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| FrameState {
                in_flight_future: None,
                descriptor_sets: vec![descriptor_set[i].clone()],
            })
            .collect::<Vec<_>>();

        let recreate_swapchain = false;

        let start_time = Instant::now();

        self.render_context = Some(RenderContext {
            swapchain,
            pipeline,
            viewport,
            recreate_swapchain,
            frames,
            current_frame: 0,
            start_time,
        });
        Ok(())
    }

    fn on_update(&mut self) -> Result<()> {
        let rcx = match self.render_context.as_mut() {
            Some(rcx) => rcx,
            None => {
                return Err(anyhow!("Render context not initialized"));
            }
        };

        let is_minimized = self.winit_window.is_minimized();
        let window_size = self.winit_window.inner_size();

        let _span_draw_frame = span!(
            Level::INFO,
            "VulkanRenderer::draw_frame",
            FrameIndex = rcx.current_frame
        )
        .entered();

        if is_minimized.is_some_and(|minimized| minimized)
            || window_size.width == 0
            || window_size.height == 0
        {
            // If the window is minimized, we skip rendering this frame.
            thread::sleep(Duration::from_millis(50));
            rcx.recreate_swapchain = true;
            return Ok(());
        }

        // Whenever the window resizes we need to recreate everything dependent on the
        // window size. In this example that includes the swapchain, the framebuffers and
        // the dynamic state viewport.
        if rcx.recreate_swapchain {
            info!(
                "Recreating swapchain for new window size: {:?}",
                window_size
            );
            rcx.swapchain.recreate(window_size.into())?;
            self.resources
                .create_depth_resources(rcx.swapchain.extent)?;
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) = match rcx
            .swapchain
            .acquire_next_image()
            .map_err(Validated::unwrap)
        {
            Ok(r) => r,
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                return Ok(());
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        if suboptimal {
            info!("Swapchain is suboptimal; recreating");
            rcx.recreate_swapchain = true;
            return Ok(());
        }

        rcx.update_uniform_buffer(
            self.resources
                .get_uniform_buffer(rcx.current_frame)
                .with_context(|| "Uniform buffer not found")?,
        )
        .with_context(|| "Failed to update uniform buffer")?;

        match rcx.build_command_buffer(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
            self.resources.get_depth_resources()?,
            image_index,
        ) {
            Ok(builder) => {
                let mut active_frame = ActiveFrame {
                    rcx,
                    resources: &self.resources,
                    builder: Some(builder),
                    image_index,
                    acquire_future: Some(acquire_future.boxed()),
                };
                active_frame.draw().with_context(|| "Failed to draw mesh")?;
                active_frame
                    .execute_command_buffer(&self.graphics_queue)
                    .with_context(|| "Failed to execute command buffer")?;
                Ok(())
            }
            Err(err) => Err(anyhow!("Failed to build command buffer: {}", err)),
        }
    }

    fn upload_mesh(&mut self, vertices: &[ElmVertex], indices: &[u32]) -> Result<()> {
        self.resources.upload_mesh(vertices, indices)?;
        Ok(())
    }

    fn upload_texture(
        &mut self,
        image_data: &[u8],
        width: u32,
        height: u32,
        filter: (Option<MagFilter>, Option<MinFilter>),
        wrap: (WrappingMode, WrappingMode),
    ) -> Result<()> {
        // Do mapping of filtering and wrapping modes to Vulkan
        let vk_mag_filter = match filter.0 {
            Some(MagFilter::Nearest) => Filter::Nearest,
            Some(MagFilter::Linear) | None => Filter::Linear,
        };
        let vk_min_filter = match filter.1 {
            Some(MinFilter::Nearest) => Filter::Nearest,
            Some(MinFilter::Linear) => Filter::Linear,
            _ => Filter::Linear, // Simplified for brevity
        };
        let vk_address_mode_s = match wrap.0 {
            WrappingMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
            WrappingMode::MirroredRepeat => SamplerAddressMode::MirroredRepeat,
            WrappingMode::Repeat => SamplerAddressMode::Repeat,
        };
        let vk_address_mode_t = match wrap.1 {
            WrappingMode::ClampToEdge => SamplerAddressMode::ClampToEdge,
            WrappingMode::MirroredRepeat => SamplerAddressMode::MirroredRepeat,
            WrappingMode::Repeat => SamplerAddressMode::Repeat,
        };
        self.resources.upload_texture(
            image_data,
            width,
            height,
            vk_mag_filter,
            vk_min_filter,
            [
                vk_address_mode_s,
                vk_address_mode_t,
                SamplerAddressMode::Repeat,
            ],
        )?;
        Ok(())
    }
}
