use std::{sync::Arc, time::Instant};

use anyhow::{Context, Result};
use glam::{Mat4, Vec2, Vec3};
use tracing::{debug, error, info};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::Subbuffer,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{
        Instance, InstanceCreateFlags, InstanceCreateInfo,
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
    },
    pipeline::{PipelineBindPoint, graphics::viewport::Viewport},
    swapchain::{Surface, SwapchainPresentInfo},
    sync::{GpuFuture, future::FenceSignalFuture},
};
use winit::window::Window as WinitWindow;

use crate::{
    renderer::renderer_vulkan::{
        buffers::{MyVertex, RenderMesh, UniformBufferObject, VulkanResourceManager},
        pipeline::VulkanPipeline,
        render_targets::RenderTargets,
        swapchain::VulkanSwapchain,
    },
    resource_manager::ResourceManager,
    window::Window,
};

mod buffers;
mod pipeline;
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
    _debug_callback: DebugUtilsMessenger,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    resource_manager: VulkanResourceManager,
    render_context: Option<RenderContext>,
}

struct RenderContext {
    swapchain: VulkanSwapchain,
    pipeline: VulkanPipeline,
    render_targets: RenderTargets,
    viewport: Viewport,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    recreate_swapchain: bool,
    in_flight_futures: Vec<Option<FenceSignalFuture<Box<dyn GpuFuture>>>>,
    current_frame: usize,
    start_time: Instant,
}

impl VulkanRenderer {
    pub fn new(resources: &ResourceManager) -> Result<VulkanRenderer> {
        let winit_window = resources
            .get::<Window>()
            .get_window()
            .with_context(|| "No window found")?;

        let vk_lib = VulkanLibrary::new()?;

        let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];
        let mut required_extensions = Surface::required_extensions(&winit_window)?;
        required_extensions.ext_debug_utils = true;
        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                enabled_layers: layers,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let _debug_callback = DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                    match message_severity {
                        DebugUtilsMessageSeverity::ERROR => {
                            error!(
                                "Vulkan Debug - ERROR - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::WARNING => {
                            error!(
                                "Vulkan Debug - WARNING - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::INFO => {
                            info!(
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
                            info!(
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
            Default::default(),
        ));

        let resource_manager = VulkanResourceManager::new(
            device.clone(),
            graphics_queue.clone(),
            command_buffer_allocator.clone(),
        );

        Ok(VulkanRenderer {
            winit_window,
            instance,
            _debug_callback,
            device,
            graphics_queue,
            command_buffer_allocator,
            resource_manager,
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

        let mut verts = VERTICES;
        let inds = INDICES;
        self.resource_manager.create_mesh(&mut verts, &inds)?;

        self.resource_manager
            .create_uniform_buffers(MAX_FRAMES_IN_FLIGHT)?;

        let descriptor_sets = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let ubo = self
                    .resource_manager
                    .get_uniform_buffer(i)
                    .with_context(|| format!("Uniform buffer {i} not found"))?;
                let set = DescriptorSet::new(
                    self.resource_manager.descriptor_set_allocator.clone(),
                    pipeline.layout().set_layouts()[0].clone(),
                    [WriteDescriptorSet::buffer(0, ubo)],
                    [],
                )?;
                Ok::<Arc<DescriptorSet>, anyhow::Error>(set)
            })
            .collect::<Result<Vec<_>>>()?;

        let recreate_swapchain = false;
        let in_flight_futures = (0..MAX_FRAMES_IN_FLIGHT).map(|_| None).collect();

        let start_time = Instant::now();

        self.render_context = Some(RenderContext {
            swapchain,
            pipeline,
            render_targets,
            viewport,
            descriptor_sets,
            recreate_swapchain,
            in_flight_futures,
            current_frame: 0,
            start_time,
        });
        Ok(())
    }

    pub fn draw_frame(&mut self, window: &Window) -> Result<()> {
        if window.is_minimized() || window.get_width() == 0 || window.get_height() == 0 {
            info!("Window is minimized or has zero size, skipping draw frame");
            return Ok(());
        }
        let rcx = match self.render_context.as_mut() {
            Some(rcx) => rcx,
            None => {
                error!("Render context not initialized, skipping draw frame");
                return Ok(());
            }
        };
        let window_size = self.winit_window.inner_size();
        // It is important to call this function from time to time, otherwise resources
        // will keep accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU
        // has already processed, and frees the resources that are no longer needed.
        if let Some(fence_future) = rcx.in_flight_futures[rcx.current_frame].as_mut() {
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
                return Ok(());
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        if suboptimal {
            rcx.recreate_swapchain = true;
            return Ok(());
        }

        // debug!("Acquired image index: {}", image_index);

        rcx.update_uniform_buffer(
            self.resource_manager
                .get_uniform_buffer(rcx.current_frame)
                .with_context(|| "Uniform buffer not found")?,
        )
        .with_context(|| "Failed to update uniform buffer")?;

        let mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                self.command_buffer_allocator.clone(),
                self.graphics_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        rcx.render_targets
                            .framebuffers(0)
                            .with_context(|| "No framebuffers for render pass 0")?
                            [image_index as usize]
                            .clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(rcx.pipeline.pipeline())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                rcx.pipeline.layout(),
                0,
                rcx.descriptor_sets[rcx.current_frame].clone(),
            )
            .with_context(|| "Failed to bind descriptor sets")?;

        let mesh = self
            .resource_manager
            .get_mesh(0)
            .with_context(|| "Mesh not found")?;
        rcx.draw_mesh(mesh, &mut builder)?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;

        // Build the future chain and obtain a fence future we can wait on next use of this slot.
        let execution_future = acquire_future
            .then_execute(self.graphics_queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    rcx.swapchain.swapchain.clone(),
                    image_index,
                ),
            )
            .boxed() // erase concrete type so we have a uniform storage type
            .then_signal_fence_and_flush();

        match execution_future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.in_flight_futures[rcx.current_frame] = Some(future);
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.in_flight_futures[rcx.current_frame] = None;
            }
            Err(e) => return Err(e.into()),
        }

        rcx.current_frame = (rcx.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }
}

impl RenderContext {
    pub fn draw_mesh(
        &mut self,
        mesh: &RenderMesh,
        cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        cbb.bind_vertex_buffers(0, mesh.vertex_buffer.clone())?
            .bind_index_buffer(mesh.index_buffer.clone())?;
        // We add a draw command.
        unsafe {
            cbb.draw_indexed(mesh.index_count, 1, 0, 0, 0)?;
        };
        Ok(())
    }

    pub fn update_uniform_buffer(
        &mut self,
        ubo_buffer: Subbuffer<UniformBufferObject>,
    ) -> Result<()> {
        let current_time = Instant::now();
        let elapsed = current_time.duration_since(self.start_time);

        let mut ubo = UniformBufferObject {
            model: Mat4::from_rotation_z(elapsed.as_secs_f32() * 90.0f32.to_radians()),
            view: Mat4::look_at_rh(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO, Vec3::Z),
            proj: Mat4::perspective_rh(
                45.0f32.to_radians(),
                self.viewport.extent[0] / self.viewport.extent[1],
                0.1,
                10.0,
            ),
        };
        ubo.proj.y_axis.y *= -1.0; // Invert Y coordinate for Vulkan

        *ubo_buffer.write()? = ubo;
        Ok(())
    }
}
