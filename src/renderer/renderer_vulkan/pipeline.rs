use std::sync::Arc;

use anyhow::{Result, anyhow};
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    device::Device,
    format::Format,
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
        layout::PipelineLayoutCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderStages,
};

use crate::renderer::renderer_vulkan::{
    MyVertex,
    shaders::{fs, vs},
};

pub struct VulkanPipeline {
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
}

impl VulkanPipeline {
    pub fn new(device: Arc<Device>, format: Format) -> Result<Self> {
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )?;

        let pipeline = {
            let vs = vs::load(device.clone())?
                .entry_point("main")
                .ok_or(anyhow!("No main entry point in vertex shader"))?;
            let fs = fs::load(device.clone())?
                .entry_point("main")
                .ok_or(anyhow!("No main entry point in fragment shader"))?;

            let vertex_input_state = MyVertex::per_vertex().definition(&vs)?;

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let rasterization_state = RasterizationState {
                polygon_mode: PolygonMode::Fill,
                line_width: 1.0,
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..RasterizationState::default()
            };

            let mut descriptor_set_layout_binding =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);

            descriptor_set_layout_binding.stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

            let descriptor_set_layout = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings: vec![(0, descriptor_set_layout_binding)]
                        .into_iter()
                        .collect(),
                    ..Default::default()
                },
            )?;

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![descriptor_set_layout],
                    ..Default::default()
                },
            )?;

            let subpass = Subpass::from(render_pass.clone(), 0)
                .ok_or_else(|| anyhow!("Subpass 0 not found"))?;

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(rasterization_state),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        Ok(VulkanPipeline {
            pipeline,
            render_pass,
        })
    }

    pub fn pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    pub fn render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
    }

    pub fn layout(&self) -> Arc<PipelineLayout> {
        self.pipeline.layout().clone()
    }
}
