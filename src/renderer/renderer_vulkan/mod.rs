use std::sync::Arc;

use anyhow::{Result, anyhow};
use glam::{Vec2, Vec3};
use tracing::{error, info};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        allocator::StandardCommandBufferAllocator,
    },
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
    pipeline::graphics::viewport::Viewport,
    swapchain::{Surface, SwapchainPresentInfo},
    sync::{self, GpuFuture},
};
use winit::window::Window as WinitWindow;

use crate::{
    renderer::renderer_vulkan::{
        buffers::{MyVertex, RenderMesh, VulkanAllocator},
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
    debug_callback: Option<DebugUtilsMessenger>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    allocator: VulkanAllocator,
    render_context: Option<RenderContext>,
}

struct RenderContext {
    swapchain: VulkanSwapchain,
    pipeline: VulkanPipeline,
    render_targets: RenderTargets,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanRenderer {
    pub fn new(resources: &ResourceManager) -> Result<VulkanRenderer> {
        let winit_window = resources
            .get::<Window>()
            .get_window()
            .ok_or_else(|| anyhow!("No window found"))?;

        let vk_lib = VulkanLibrary::new()?;

        let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];
        let required_extensions = Surface::required_extensions(&winit_window)?;
        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                enabled_layers: layers,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let debug_callback = DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                    match message_severity {
                        DebugUtilsMessageSeverity::ERROR => {
                            error!(
                                "Vulkan Debug Callback - ERROR - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::WARNING => {
                            error!(
                                "Vulkan Debug Callback - WARNING - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::INFO => {
                            info!(
                                "Vulkan Debug Callback - INFO - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        DebugUtilsMessageSeverity::VERBOSE => {
                            info!(
                                "Vulkan Debug Callback - VERBOSE - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                        _ => {
                            info!(
                                "Vulkan Debug Callback - UNKNOWN - {:?} - {:?}: {}",
                                message_type, message_severity, callback_data.message
                            );
                        }
                    }
                })
            }),
        )
        .ok();

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
            .ok_or_else(|| anyhow!("No suitable physical device found"))?;

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
        let graphics_queue: Arc<Queue> = queues_iter
            .next()
            .ok_or_else(|| anyhow!("No queue found"))?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let mut allocator = VulkanAllocator::new(
            device.clone(),
            graphics_queue.clone(),
            command_buffer_allocator.clone(),
        );

        // Create local mutable copies instead of taking &mut of const items.
        let mut verts = VERTICES;
        let inds = INDICES;
        allocator.create_mesh(&mut verts, &inds)?;

        Ok(VulkanRenderer {
            winit_window,
            instance,
            debug_callback,
            device,
            graphics_queue,
            command_buffer_allocator,
            allocator,
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

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.render_context = Some(RenderContext {
            swapchain,
            pipeline,
            render_targets,
            viewport,
            recreate_swapchain,
            previous_frame_end,
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
        rcx.previous_frame_end
            .as_mut()
            .ok_or_else(|| anyhow!("Previous frame could not be borrowed"))?
            .cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the
        // window size. In this example that includes the swapchain, the framebuffers and
        // the dynamic state viewport.
        if rcx.recreate_swapchain {
            rcx.swapchain.recreate(window_size.into())?;

            // Because framebuffers contains a reference to the old swapchain, we need to
            // recreate framebuffers as well.
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
                error!("Failed to acquire next image: {e}");
                return Err(anyhow!("Failed to acquire next image: {e}"));
            }
        };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

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
                            .ok_or_else(|| anyhow!("No framebuffers for render pass 0"))?
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
            .bind_pipeline_graphics(rcx.pipeline.pipeline())?;

        let mesh = self
            .allocator
            .get_mesh(0)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Mesh not found"))?;
        rcx.draw_mesh(mesh, &mut builder)?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;

        let future = rcx
            .previous_frame_end
            .take()
            .ok_or_else(|| anyhow!("Previous frame could not be taken"))?
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    rcx.swapchain.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
                Ok(())
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
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
}
