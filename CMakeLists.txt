cmake_minimum_required(VERSION 3.21)

# ── Project ──────────────────────────────────────────────────────────────────
project(MoonAI
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(MOONAI_BUILD_TESTS  "Build unit tests"                 ON)

# ── CUDA auto-detection ───────────────────────────────────────────────────────
# Not a user option — enabled automatically when nvcc is present.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MOONAI_ENABLE_CUDA ON)
    message(STATUS "CUDA auto-detected: ${CMAKE_CUDA_COMPILER}")
else()
    set(MOONAI_ENABLE_CUDA OFF)
    message(STATUS "CUDA not found — building CPU-only.")
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(SFML 3 COMPONENTS Graphics Window System REQUIRED)

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP)

# ── Subsystem libraries ─────────────────────────────────────────────────────
add_subdirectory(src/core)
add_subdirectory(src/simulation)
add_subdirectory(src/evolution)
add_subdirectory(src/data)
add_subdirectory(src/visualization)

if(MOONAI_ENABLE_CUDA)
    add_subdirectory(src/gpu)
endif()

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(moonai src/main.cpp)

target_link_libraries(moonai PRIVATE
    moonai_core
    moonai_simulation
    moonai_evolution
    moonai_data
)

target_link_libraries(moonai PRIVATE moonai_visualization)

if(MOONAI_ENABLE_CUDA)
    # moonai_gpu has no dependency on moonai_evolution (no circular dep):
    # the network-packing logic lives in evolution_manager.cpp (moonai_evolution side).
    target_link_libraries(moonai_evolution PRIVATE moonai_gpu)
    target_compile_definitions(moonai_evolution PRIVATE MOONAI_ENABLE_CUDA)

    target_link_libraries(moonai PRIVATE moonai_gpu)
    target_compile_definitions(moonai PRIVATE MOONAI_ENABLE_CUDA)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(MOONAI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
