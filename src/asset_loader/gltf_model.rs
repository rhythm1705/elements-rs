use crate::core::vertex::{ElmVec2, ElmVec3, ElmVertex};
use anyhow::{Context, anyhow};
use assets_manager::asset::Gltf;
use assets_manager::{Asset, AssetCache, BoxedError, SharedString};
use glam::{Vec2, Vec3};
use gltf::image::Format;
use gltf::texture::{MagFilter, MinFilter, WrappingMode};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Primitive {
    pub vertices: Vec<ElmVertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug)]
pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

#[derive(Debug)]
pub struct Image {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: Format,
}

#[derive(Debug)]
pub struct Sampler {
    pub mag_filter: Option<MagFilter>,
    pub min_filter: Option<MinFilter>,
    pub wrap_s: WrappingMode,
    pub wrap_t: WrappingMode,
}

#[derive(Debug)]
pub struct Texture {
    pub image: Option<usize>,
    pub sampler: Sampler,
}

#[derive(Debug)]
pub struct Material {
    pub textures: Vec<u32>,
}

#[derive(Debug)]
pub struct Node {
    pub mesh_id: Option<usize>,
    pub children: Vec<usize>, // Indices of child nodes in the Scene's nodes vector
}

#[derive(Debug)]
pub struct Scene {
    pub nodes: Vec<usize>, // Only root node indices
}

#[derive(Debug)]
pub struct GltfModel {
    pub scenes: Vec<Scene>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
}

impl Asset for GltfModel {
    fn load(cache: &AssetCache, id: &SharedString) -> Result<Self, BoxedError> {
        let handle = cache
            .load::<Gltf>(id)
            .with_context(|| "Gltf could not be loaded.")?;
        let gltf = handle.read();

        let mut scenes = Vec::new();
        for scene in gltf.document.scenes() {
            let mut root_nodes = Vec::new();
            for node in scene.nodes() {
                root_nodes.push(node.index());
            }
            scenes.push(Scene { nodes: root_nodes });
        }

        let mut nodes = Vec::new();
        for node in gltf.document.nodes() {
            let mesh_id = node.mesh().map(|mesh| mesh.index());
            let mut child_indices = Vec::new();
            for child in node.children() {
                child_indices.push(child.index());
            }
            nodes.push(Node {
                mesh_id,
                children: child_indices,
            });
        }

        let mut meshes = Vec::new();
        for mesh in gltf.document.meshes() {
            let mut primitives = Vec::new();
            for primitive in mesh.primitives() {
                let reader =
                    primitive.reader(|buffer| Some(gltf.get_buffer_by_index(buffer.index())));
                let indices: Vec<u32> = reader
                    .read_indices()
                    .ok_or(anyhow!("No indices in mesh"))?
                    .into_u32()
                    .collect();
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or(anyhow!("No positions in mesh"))?
                    .collect();
                let tex_coords: Option<Vec<[f32; 2]>> =
                    reader.read_tex_coords(0).map(|tc| tc.into_f32().collect());

                let mut unique_vertices = HashMap::<ElmVertex, u32>::new();
                let mut vertices: Vec<ElmVertex> = Vec::new();
                let mut remapped_indices: Vec<u32> = Vec::with_capacity(indices.len());

                for &i in &indices {
                    let position = ElmVec3::from(Vec3::from(positions[i as usize]));
                    let tex_coord = if let Some(ref tcs) = tex_coords {
                        ElmVec2::from(Vec2::from(tcs[i as usize]))
                    } else {
                        ElmVec2::from(Vec2::new(0.0, 0.0))
                    };
                    let color = ElmVec3::from(Vec3::new(1.0, 1.0, 1.0)); // Default white color

                    let vertex = ElmVertex {
                        position,
                        color,
                        tex_coord,
                    };

                    let index = *unique_vertices.entry(vertex).or_insert_with(|| {
                        let new_index = vertices.len() as u32;
                        vertices.push(vertex);
                        new_index
                    });
                    remapped_indices.push(index);
                }

                if vertices.is_empty() {
                    continue; // Skip empty meshes
                }

                primitives.push(Primitive {
                    vertices,
                    indices: remapped_indices,
                });
            }
            meshes.push(Mesh { primitives });
        }

        let mut images = Vec::new();
        for image in gltf.document.images() {
            let image_data = gltf.get_image(&image).to_rgba8();
            let (width, height) = image_data.dimensions();
            images.push(Image {
                pixels: image_data.into_raw(),
                width,
                height,
                format: Format::R8G8B8A8,
            });
        }

        let mut textures = Vec::new();
        for texture in gltf.document.textures() {
            let sampler = texture.sampler();
            let sampler = Sampler {
                mag_filter: sampler.mag_filter(),
                min_filter: sampler.min_filter(),
                wrap_s: sampler.wrap_s(),
                wrap_t: sampler.wrap_t(),
            };
            let image = texture.source().index();
            textures.push(Texture {
                image: Some(image),
                sampler,
            });
        }

        Ok(GltfModel {
            scenes,
            nodes,
            meshes,
            images,
            textures,
            materials: Vec::new(),
        })
    }
}
