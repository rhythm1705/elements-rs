use crate::renderer::renderer_vulkan::render_context::FrameState;
pub(crate) use crate::{
    renderer::renderer_vulkan::{
        buffers::{MyVertex, VulkanResourceManager},
        pipeline::VulkanPipeline,
        render_context::{ActiveFrame, RenderContext},
        render_targets::RenderTargets,
        swapchain::VulkanSwapchain,
    },
    resource_manager::ResourceManager,
    window::Window,
};
use anyhow::{Context, Result, anyhow};
use glam::{Vec2, Vec3};
use std::{sync::Arc, time::Instant};
#[cfg(debug_assertions)]
use tracing::debug;
use tracing::info;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocatorCreateInfo;
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

mod buffers;
mod pipeline;
mod render_context;
mod render_targets;
mod shaders;
mod swapchain;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const VERTICES: [MyVertex; 4] = [
    MyVertex {
        position: Vec2::new(-0.5, -0.5),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    MyVertex {
        position: Vec2::new(0.5, -0.5),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    MyVertex {
        position: Vec2::new(0.5, 0.5),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    MyVertex {
        position: Vec2::new(-0.5, 0.5),
        color: Vec3::new(1.0, 1.0, 1.0),
    },
];

const INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub struct VulkanRenderer {
    winit_window: Arc<WinitWindow>,
    instance: Arc<Instance>,
    #[cfg(debug_assertions)]
    _debug_callback: DebugUtilsMessenger,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    resources: VulkanResourceManager,
    render_context: Option<RenderContext>,
}

impl VulkanRenderer {
    pub fn new(resources: &ResourceManager) -> Result<VulkanRenderer> {
        let winit_window = resources.get::<Window>().get_winit_window();

        let vk_lib = VulkanLibrary::new()?;

        let enable_validation = cfg!(debug_assertions);

        let layers = if enable_validation {
            vec!["VK_LAYER_KHRONOS_validation".to_owned()]
        } else {
            Vec::new()
        };
        let mut required_extensions = Surface::required_extensions(&winit_window)?;
        if enable_validation {
            required_extensions.ext_debug_utils = true;
        }
        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                enabled_layers: layers,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

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
        .with_context(|| "Failed to create debug callback")?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
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
            .with_context(|| "No suitable physical device found")?;

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

                ..Default::default()
            },
        )?;
        let graphics_queue: Arc<Queue> = queues_iter.next().with_context(|| "No queue found")?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let resources = VulkanResourceManager::new(
            device.clone(),
            graphics_queue.clone(),
            command_buffer_allocator.clone(),
        );

        Ok(VulkanRenderer {
            winit_window,
            instance,
            #[cfg(debug_assertions)]
            _debug_callback,
            device,
            graphics_queue,
            command_buffer_allocator,
            resources,
            render_context: None,
        })
    }

    pub fn initialize_render_context(&mut self) -> Result<()> {
        let surface = Surface::from_window(self.instance.clone(), self.winit_window.clone())?;
        let window_size = self.winit_window.inner_size();

        let swapchain =
            VulkanSwapchain::new(self.device.clone(), surface.clone(), window_size.into())?;

        let pipeline = VulkanPipeline::new(self.device.clone(), swapchain.format)?;

        let mut render_targets = RenderTargets::new(swapchain.images.clone());

        render_targets.rebuild_for_pass(0, &pipeline.render_pass())?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let mut vertices = VERTICES;
        let indices = INDICES;
        self.resources.create_mesh(&mut vertices, &indices)?;

        self.resources
            .create_uniform_buffers(MAX_FRAMES_IN_FLIGHT)?;

        let descriptor_sets = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let ubo = self
                    .resources
                    .get_uniform_buffer(i)
                    .with_context(|| format!("Uniform buffer {i} not found"))?;
                let set = DescriptorSet::new(
                    self.resources.descriptor_set_allocator.clone(),
                    pipeline.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::buffer(0, ubo)],
                    [],
                )?;
                Ok::<Arc<DescriptorSet>, anyhow::Error>(set)
            })
            .collect::<Result<Vec<_>>>()?;

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| FrameState {
                in_flight_future: None,
                descriptor_set: descriptor_sets[i].clone(),
            })
            .collect::<Vec<_>>();

        let recreate_swapchain = false;

        let start_time = Instant::now();

        self.render_context = Some(RenderContext {
            swapchain,
            pipeline,
            render_targets,
            viewport,
            recreate_swapchain,
            frames,
            current_frame: 0,
            start_time,
        });
        Ok(())
    }

    pub fn draw_frame(&'_ mut self) -> Result<()> {
        let is_minimized = self.winit_window.is_minimized();
        let window_size = self.winit_window.inner_size();

        if is_minimized.is_none_or(|e| e) || window_size.width == 0 || window_size.height == 0 {
            info!("Window is minimized or has zero size, skipping draw frame");
            return Err(anyhow!("Window is minimized or has zero size"));
        }

        let rcx = match self.render_context.as_mut() {
            Some(rcx) => rcx,
            None => {
                return Err(anyhow!("Render context not initialized"));
            }
        };

        // It is important to call this function from time to time, otherwise resources
        // will keep accumulating, and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU
        // has already processed, and frees the resources that are no longer needed.
        if let Some(fence_future) = rcx.frames[rcx.current_frame].in_flight_future.as_mut() {
            fence_future.wait(None)?; // ensure safe reuse of this slot's UBO
            fence_future.cleanup_finished();
        }

        // Whenever the window resizes we need to recreate everything dependent on the
        // window size. In this example that includes the swapchain, the framebuffers and
        // the dynamic state viewport.
        if rcx.recreate_swapchain {
            rcx.swapchain.recreate(window_size.into())?;
            rcx.render_targets
                .replace_images(rcx.swapchain.images.clone());
            rcx.render_targets
                .rebuild_for_pass(0, &rcx.pipeline.render_pass())?;
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
                return Err(anyhow!("Swapchain out of date"));
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        if suboptimal {
            rcx.recreate_swapchain = true;
            return Err(anyhow!("Swapchain suboptimal"));
        }

        // debug!("Acquired image index: {}", image_index);

        rcx.update_uniform_buffer(
            self.resources
                .get_uniform_buffer(rcx.current_frame)
                .with_context(|| "Uniform buffer not found")?,
        )
        .with_context(|| "Failed to update uniform buffer")?;

        if let Ok(builder) = rcx.build_command_buffer(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.clone(),
            image_index,
        ) {
            let mut active_frame = ActiveFrame {
                rcx,
                resources: &self.resources,
                builder: Some(builder),
                image_index,
                acquire_future: Some(acquire_future.boxed()),
                _finished: false,
            };
            active_frame
                .draw_mesh(0)
                .with_context(|| "Failed to draw mesh")?;
            active_frame
                .execute_command_buffer(&self.graphics_queue.clone())
                .with_context(|| "Failed to execute command buffer")?;
            Ok(())
        } else {
            Err(anyhow!("Failed to build command buffer"))
        }
    }
}
