# Elements

A small, hobby game engine built to learn Rust and graphics programming using Vulkan. It uses winit for windowing/event handling and Vulkano as the Vulkan wrapper. Expect rapid changes and breaking refactors as this is a learning project.

## Status
- Work in progress / experimental
- Currently focused on Windows

## Getting started

### Prerequisites
- Rust toolchain (install via https://rustup.rs)
- A GPU and drivers with Vulkan support
- Optional but recommended: Vulkan SDK (for validation layers and tools)
  - https://vulkan.lunarg.com/

### Build and run
```pwsh
# from the repo root
cargo run
```

If you have multiple Rust toolchains or targets, consider:
```pwsh
cargo run --release
```

You might have to setup environment variable for Vulkan SDK or provide a native shaderc library. 

## Tech stack
- Rust
- Vulkan (via Vulkano)
- winit (windowing/event loop)
- tracing (logging)

## Project goals
- Learn idiomatic Rust by building engine subsystems (windowing, input, renderer, resources)
- Explore Vulkan fundamentals: instance/device setup, swapchains, render passes, pipelines, and synchronization

## Notes
- Logging is enabled via `tracing` and pretty formatting; check the console for diagnostics
- The codebase is evolving; APIs and structure may change frequently
