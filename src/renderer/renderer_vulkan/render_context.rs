use crate::renderer::renderer_vulkan::resources::VulkanResources;
use crate::renderer::renderer_vulkan::{
    MAX_FRAMES_IN_FLIGHT, pipeline::VulkanPipeline, resources::UniformBufferObject,
    swapchain::VulkanSwapchain,
};
use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use std::{sync::Arc, time::Instant};
use tracing::error;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::Queue;
use vulkano::format::ClearValue;
use vulkano::image::ImageLayout::DepthAttachmentOptimal;
use vulkano::image::view::ImageView;
use vulkano::pipeline::PipelineBindPoint;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::{
    Validated, VulkanError,
    buffer::Subbuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::DescriptorSet,
    pipeline::graphics::viewport::Viewport,
    sync::{GpuFuture, future::FenceSignalFuture},
};

pub struct RenderContext {
    pub swapchain: VulkanSwapchain,
    pub pipeline: VulkanPipeline,
    pub viewport: Viewport,
    pub recreate_swapchain: bool,
    pub frames: Vec<FrameState>,
    pub current_frame: usize,
    pub start_time: Instant,
}

pub struct FrameState {
    pub in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
    pub descriptor_sets: Vec<Arc<DescriptorSet>>,
}

impl RenderContext {
    pub fn update_uniform_buffer(
        &mut self,
        ubo_buffer: Subbuffer<UniformBufferObject>,
    ) -> Result<()> {
        let current_time = Instant::now();
        let elapsed = current_time.duration_since(self.start_time);

        let mut ubo = UniformBufferObject {
            model: Mat4::from_rotation_z(elapsed.as_secs_f32() * 90.0f32.to_radians()),
            view: Mat4::look_at_rh(Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO, Vec3::Z),
            proj: Mat4::perspective_rh(
                45.0f32.to_radians(),
                self.viewport.extent[0] / self.viewport.extent[1],
                0.5,
                20.0,
            ),
        };
        ubo.proj.y_axis.y *= -1.0; // Invert Y coordinate for Vulkan

        *ubo_buffer.write()? = ubo;
        Ok(())
    }

    pub fn build_command_buffer(
        &mut self,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        graphics_queue: Arc<Queue>,
        depth_image_view: Arc<ImageView>,
        image_index: u32,
    ) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let mut builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                graphics_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?;

        let clear_color = ClearValue::Float([0.0, 0.0, 0.0, 1.0]);
        let clear_depth = ClearValue::DepthStencil((1.0, 0));

        let color_attachments = vec![Some(RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: Some(clear_color),
            ..RenderingAttachmentInfo::image_view(
                self.swapchain.image_views[image_index as usize].clone(),
            )
        })];

        let depth_attachment = Some(RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::DontCare,
            clear_value: Some(clear_depth),
            image_layout: DepthAttachmentOptimal,
            ..RenderingAttachmentInfo::image_view(depth_image_view.clone())
        });

        builder
            .begin_rendering(RenderingInfo {
                render_area_extent: self.swapchain.extent,
                layer_count: 1,
                color_attachments,
                depth_attachment,
                ..Default::default()
            })
            .with_context(|| "Begin rendering")?
            .bind_pipeline_graphics(self.pipeline.pipeline())?
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout(),
                0,
                self.frames[self.current_frame].descriptor_sets.clone(),
            )
            .with_context(|| "Bind descriptor sets")?;

        Ok(builder)
    }
}

pub struct ActiveFrame<'a> {
    pub rcx: &'a mut RenderContext,
    pub resources: &'a VulkanResources,
    pub builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    pub image_index: u32,
    pub acquire_future: Option<Box<dyn GpuFuture>>,
}

impl<'a> ActiveFrame<'a> {
    pub fn draw(&mut self) -> Result<()> {
        for mesh in &self.resources.meshes {
            if let Some(ref mut builder) = self.builder {
                builder
                    .bind_vertex_buffers(0, mesh.vertex_buffer.clone())?
                    .bind_index_buffer(mesh.index_buffer.clone())?;
                // We add a draw command.
                unsafe {
                    builder.draw_indexed(mesh.index_count, 1, 0, 0, 0)?;
                };
            } else {
                return Err(anyhow::anyhow!("Command buffer builder not initialized"));
            }
        }

        Ok(())
    }

    pub fn execute_command_buffer(&mut self, graphics_queue: &Arc<Queue>) -> Result<()> {
        let mut builder = self
            .builder
            .take()
            .ok_or_else(|| anyhow::anyhow!("Command buffer builder not initialized"))?;
        builder.end_rendering()?;

        let command_buffer = builder.build()?;

        // Build the future chain and obtain a fence future we can wait on next use of this slot.
        let execution_future = self
            .acquire_future
            .take()
            .ok_or_else(|| anyhow::anyhow!("Acquire future not complete"))?
            .then_execute(graphics_queue.clone(), command_buffer)?
            .then_swapchain_present(
                graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.rcx.swapchain.swapchain.clone(),
                    self.image_index,
                ),
            )
            .boxed() // erase concrete type so we have a uniform storage type
            .then_signal_fence_and_flush();

        match execution_future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.rcx.frames[self.rcx.current_frame].in_flight_future = Some(future);
            }
            Err(VulkanError::OutOfDate) => {
                self.rcx.recreate_swapchain = true;
                self.rcx.frames[self.rcx.current_frame].in_flight_future = None;
            }
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }
}

impl Drop for ActiveFrame<'_> {
    fn drop(&mut self) {
        if let Some(fence_future) = self.rcx.frames[self.rcx.current_frame]
            .in_flight_future
            .as_mut()
        {
            match fence_future.wait(None) {
                Ok(()) => {
                    // Fence successfully waited: we can safely clean up and advance to the next frame.
                    fence_future.cleanup_finished();
                    self.rcx.current_frame = (self.rcx.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
                }
                Err(e) => {
                    error!("Failed to wait for fence future: {:?}", e);
                }
            }
        }
    }
}
