use std::{error::Error, sync::Arc};

use tracing::{error, info};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::{vertex_input::Vertex, viewport::Viewport},
    swapchain::{Surface, SwapchainPresentInfo},
    sync::{self, GpuFuture},
};
use winit::window::Window as WinitWindow;

use crate::{
    renderer::renderer_vulkan::{
        pipeline::VulkanPipeline, render_targets::RenderTargets, swapchain::VulkanSwapchain,
    },
    resource_manager::ResourceManager,
    window::Window,
};

mod pipeline;
mod render_targets;
mod shaders;
mod swapchain;

pub struct VulkanRenderer {
    winit_window: Arc<WinitWindow>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
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
    pub fn new(resources: &ResourceManager) -> Result<VulkanRenderer, Box<dyn Error>> {
        let winit_window = resources
            .get::<Window>()
            .get_window()
            .ok_or("No window found")?;

        let vk_lib = VulkanLibrary::new()?;

        let required_extensions = Surface::required_extensions(&winit_window)?;
        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create instance");

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
            .ok_or("No suitable physical device found")?;

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
        let graphics_queue: Arc<Queue> = queues_iter.next().ok_or("No queue found")?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let vertices = [
            MyVertex {
                position: [-0.5, -0.25],
            },
            MyVertex {
                position: [0.0, 0.5],
            },
            MyVertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )?;

        Ok(VulkanRenderer {
            winit_window,
            instance,
            device,
            graphics_queue,
            command_buffer_allocator,
            vertex_buffer,
            render_context: None,
        })
    }

    pub fn initialize_render_context(&mut self) -> Result<(), Box<dyn Error>> {
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

    pub fn draw_frame(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
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
            .ok_or("Previous frame could not be borrowed")?
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

        // Before we can draw on the output, we have to *acquire* an image from the
        // swapchain. If no image is available (which happens if you submit draw commands
        // too quickly), then the function will block. This operation returns the index of
        // the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional
        // timeout after which the function call will return an error.
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
                return Err(Box::new(e));
            }
        };

        // `acquire_next_image` can be successful, but suboptimal. This means that the
        // swapchain image will still work, but it may not display correctly. With some
        // drivers this can be when the window resizes, but it may not cause the swapchain
        // to become out of date.
        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        // In order to draw, we have to record a *command buffer*. The command buffer
        // object holds the list of commands that are going to be executed.
        //
        // Recording a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to
        // be optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The
        // command buffer will only be executable on that given queue family.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            // Before we can draw, we have to *enter a render pass*.
            .begin_render_pass(
                RenderPassBeginInfo {
                    // A list of values to clear the attachments with. This list contains
                    // one item for each attachment in the render pass. In this case, there
                    // is only one attachment, and we clear it with a blue color.
                    //
                    // Only attachments that have `AttachmentLoadOp::Clear` are provided
                    // with clear values, any others should use `None` as the clear value.
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                    ..RenderPassBeginInfo::framebuffer(
                        rcx.render_targets
                            .framebuffers(0)
                            .ok_or("No framebuffers for render pass 0")?
                            [image_index as usize]
                            .clone(),
                    )
                },
                SubpassBeginInfo {
                    // The contents of the first (and only) subpass. This can be either
                    // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                    // advanced and is not covered here.
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            // We are now inside the first subpass of the render pass.
            //
            // TODO: Document state setting and how it affects subsequent draw commands.
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(rcx.pipeline.pipeline())?
            .bind_vertex_buffers(0, self.vertex_buffer.clone())?;

        // We add a draw command.
        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }?;

        builder
            // We leave the render pass. Note that if we had multiple subpasses we could
            // have called `next_subpass` to jump to the next subpass.
            .end_render_pass(Default::default())?;

        // Finish recording the command buffer by calling `end`.
        let command_buffer = builder.build()?;

        let future = rcx
            .previous_frame_end
            .take()
            .ok_or("Previous frame could not be taken")?
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)?
            // The color output is now expected to contain our triangle. But in order to
            // show it on the screen, we have to *present* the image by calling
            // `then_swapchain_present`.
            //
            // This function does not actually present the image immediately. Instead it
            // submits a present command at the end of the queue. This means that it will
            // only be presented once the GPU has finished executing the command buffer
            // that draws the triangle.
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
            Err(e) => {
                Err(Box::new(e))
                // previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
