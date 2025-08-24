use std::sync::Arc;

use tracing::info;
use vulkano::{
    VulkanLibrary,
    device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
};

use crate::{resource_manager::ResourceManager, window::Window};

pub struct Renderer {
    rcx: Option<RenderContext>,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer { rcx: None }
    }

    pub fn run(&mut self, resources: &ResourceManager) {
        self.rcx = Some(RenderContext::new(resources));
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

struct RenderContext {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queues: Vec<Arc<vulkano::device::Queue>>,
    surface: Option<Arc<Surface>>,
}

impl RenderContext {
    fn new(resources: &ResourceManager) -> RenderContext {
        let window = resources
            .get::<Window>()
            .get_window()
            .expect("No window to draw");

        let vk_lib = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let required_extensions = Surface::required_extensions(&window).unwrap();
        let instance = Instance::new(
            vk_lib,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        for physical_device in instance.enumerate_physical_devices().unwrap() {
            info!(
                "Vulkan supported device found: {}",
                physical_device.properties().device_name
            );
        }
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");
        info!(
            "Selected device: {}",
            physical_device.properties().device_name
        );

        for family in physical_device.queue_family_properties() {
            info!(
                "Found a queue family with {:?} queue(s)",
                family.queue_count
            );
        }

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        let (device, queues_iter) = Device::new(
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queues: Vec<Arc<vulkano::device::Queue>> = queues_iter.collect();

        let surface = Some(
            Surface::from_window(instance.clone(), window)
                .expect("Failed to create surface from window"),
        );

        RenderContext {
            instance,
            device,
            queues,
            surface,
        }
    }
}
