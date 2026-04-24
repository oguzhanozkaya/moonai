I inspected the current boundaries first. The migration seam is clear:
- Host/runtime logic is in C++: src/main.cpp, src/app/app.cpp, src/core/*, src/simulation/simulation.cpp, src/evolution/* except inference_cache.cu, src/metrics/*, src/visualization/*.
- CUDA is isolated enough to preserve: src/simulation/buffers.cu, src/simulation/batch.cu, src/evolution/inference_cache.cu.
- SFML is fully contained in src/visualization/*.
- Profiler code is duplicated through MOONAI_BUILD_PROFILER branches in CMakeLists.txt, src/*/CMakeLists.txt, src/profiler_main.cpp, and src/core/profiler_macros.hpp.
Assumptions
- Linux-first migration.
- config.lua stays the config format.
- Output schema stays unchanged so Python analysis keeps working.
- We want behavioral parity first, not perfect bit-for-bit parity at all costs.
Recommendation
Do not do this as a big-bang rewrite. Final state should be all-Rust host code, but implementation should be phased with a temporary Rust/CUDA bridge. The first Rust target should be headless mode. GUI comes after that.
Target Architecture
Use Cargo as the primary build and isolate CUDA behind one small unsafe boundary.
Suggested layout:
Cargo.toml
crates/
  moonai_cuda_sys/
    build.rs
    src/lib.rs
    cuda/
      buffers.cu
      batch.cu
      inference_cache.cu
      ffi.cuh
src/
  main.rs
  cli.rs
  app.rs
  config.rs
  lua_config.rs
  random.rs
  types.rs
  state.rs
  simulation/
  evolution/
  metrics/
  ui/
  render/
tests/
moonai_cuda_sys should:
- compile the .cu files in build.rs
- expose a narrow extern "C" ABI
- hide all raw CUDA ownership behind opaque handles
Do not use bindgen first. Manual #[repr(C)] structs plus opaque pointers are safer here.
CUDA Boundary
Refactor the .cu side to export C ABI wrappers only. Keep device memory, pinned host memory, streams, and kernel launch details inside .cu.
Expose handles like:
- MoonaiCudaBatch
- MoonaiCudaInferenceCache
Expose plain ABI structs for:
- StepParams
- compiled network views
- error/result codes
Do not expose internal C++ classes to Rust. simulation::Batch and evolution::InferenceCache should disappear from the host side and become Rust wrappers over opaque CUDA handles.
UI Replacement
Do not port the SFML renderer literally.
Use:
- winit for event loop/window
- wgpu for world rendering
- egui for overlay, controls, charts, and NN topology panel
Important constraint: current defaults are very large (24k predators, 96k prey, 240k food in src/core/config.hpp). That means:
- world entities must be drawn with wgpu instancing
- egui should only draw panels and lightweight overlays
- do not try to render the whole world through egui shapes
Migration Plan
1. Remove profiler support first.
   Delete src/profiler_main.cpp, remove MOONAI_BUILD_PROFILER branches, remove profiler-specific CMake targets and recipes, and strip MOONAI_PROFILE_SCOPE usage down to no-ops or remove it entirely. This reduces the port surface immediately.
2. Add Cargo workspace and moonai_cuda_sys.
   Introduce Cargo.toml, set up build.rs, and compile the existing .cu files from Cargo. Keep CMake temporarily only as a fallback until Rust headless runs.
3. Define the Rust/CUDA ABI.
   Create ffi.cuh plus matching Rust #[repr(C)] definitions. Convert the CUDA code to export C ABI functions for buffer lifecycle, capacity management, sensor build, inference launch, post-inference step, sync, and error reporting.
4. Port core/runtime infrastructure to Rust.
   Port types, config, config validation, JSON snapshot writing, Lua loading, CLI parsing, logger setup, and RNG. Keep config.lua semantics identical, including injected moonai_defaults.
5. Port the simulation state model to Rust.
   Port Food, AgentRegistry, AppState, compaction, deterministic respawn, and metrics state. Keep the same SoA layout with Vec<T> fields to preserve iteration order and keep the CUDA handoff simple.
6. Port evolution CPU logic to Rust.
   Port Genome, Mutation, Crossover, Species, NeuralNetwork, NetworkCache, and the evolution manager. Port the existing tests from tests/test_evolution.cpp first and make them pass in Rust before wiring the full runtime.
7. Port the headless step pipeline.
   Recreate the current App::step flow from src/app/app.cpp and src/simulation/simulation.cpp in Rust:
   prepare_step -> inference -> resolve_step -> post_step -> metrics.
   Keep logging/output behavior the same.
8. Make headless Rust the first working milestone.
   Before touching the GUI, Rust should successfully:
   - validate config.lua
   - run a seeded headless experiment
   - produce config.json, stats.csv, species.csv, genomes.json
9. Replace SFML with winit + wgpu + egui.
   Port input semantics from src/visualization/visualization_manager.cpp: pause, step, speed changes, zoom, pan, selection, screenshot if wanted. Rebuild the overlay in egui instead of low-level manual drawing. Render world geometry with wgpu.
10. Remove remaining C++ host code and CMake app targets.
   Once Rust headless and GUI are working, delete src/main.cpp, src/app/*, src/core/*.cpp, src/simulation/simulation.cpp, src/evolution/*.cpp, src/metrics/*.cpp, src/visualization/*, and the old CMake graph. Keep only Rust plus CUDA.
Verification Gates
Use these as hard checkpoints:
1. Core parity:
   cargo test
   Ported evolution/config tests must pass.
2. Build parity:
   cargo build
   Rust plus CUDA must build without CMake.
3. Config parity:
   cargo run -- --validate config.lua
   Must match current behavior.
4. Headless runtime parity:
   Run a small seeded config in both implementations and compare stats.csv/species.csv trends and output shape.
5. GUI smoke test:
   Open window, pan/zoom/select/pause/step/speed controls work.
