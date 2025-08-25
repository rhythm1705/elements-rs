use std::any::{Any, TypeId};
use std::collections::HashMap;

use tracing::{error, info};

/// A generic container for storing and retrieving shared data of any type.
pub struct ResourceManager {
    resources: HashMap<TypeId, Box<dyn Any>>,
}

impl ResourceManager {
    pub fn new() -> Self {
        ResourceManager {
            resources: HashMap::new(),
        }
    }

    pub fn add<T: 'static>(&mut self, resource: T) {
        let type_id = TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();

        let boxed_resource = Box::new(resource);

        self.resources.insert(type_id, boxed_resource);

        info!("Added resource {type_name}");
    }

    pub fn get<T: 'static>(&self) -> &T {
        let type_id = TypeId::of::<T>();
        self.resources
            .get(&type_id)
            .and_then(|boxed| boxed.downcast_ref::<T>())
            .unwrap_or_else(|| {
                error!("Resource of type {:?} not found", type_id);
                panic!("Resource of type {:?} not found", type_id)
            })
    }

    pub fn get_mut<T: 'static>(&mut self) -> &mut T {
        let type_id = TypeId::of::<T>();
        self.resources
            .get_mut(&type_id)
            .and_then(|boxed| boxed.downcast_mut::<T>())
            .unwrap_or_else(|| {
                error!("Resource of type {:?} not found", type_id);
                panic!("Resource of type {:?} not found", type_id)
            })
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}
