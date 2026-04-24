cmake_minimum_required(VERSION 3.21)

project(moonai-ffi CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80)

# Find dependencies via vcpkg
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# MoonAI source directories
set(MOONAI_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(MOONAI_CORE_DIR ${MOONAI_SRC_DIR}/core)
set(MOONAI_SIM_DIR ${MOONAI_SRC_DIR}/simulation)
set(MOONAI_EVO_DIR ${MOONAI_SRC_DIR}/evolution)

# Compile definitions
add_compile_definitions(__CUDACC__)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MOONAI_SRC_DIR}
    ${MOONAI_CORE_DIR}
    ${MOONAI_SIM_DIR}
    ${MOONAI_EVO_DIR}
)

# Source files for the FFI library
set(FFI_SOURCES
    src_cpp/random.cpp
    src_cpp/simulation.cpp
)

# MoonAI sources needed for FFI
set(MOONAI_SOURCES
    ${MOONAI_CORE_DIR}/app_state.cpp
    ${MOONAI_CORE_DIR}/random.cpp
    ${MOONAI_CORE_DIR}/config.cpp
    ${MOONAI_CORE_DIR}/lua_runtime.cpp
    ${MOONAI_CORE_DIR}/types.cpp
    ${MOONAI_EVO_DIR}/genome.cpp
    ${MOONAI_EVO_DIR}/mutation.cpp
    ${MOONAI_EVO_DIR}/crossover.cpp
    ${MOONAI_EVO_DIR}/species.cpp
    ${MOONAI_EVO_DIR}/neural_network.cpp
    ${MOONAI_EVO_DIR}/network_cache.cpp
    ${MOONAI_EVO_DIR}/evolution_manager.cpp
    ${MOONAI_SIM_DIR}/simulation.cpp
)

# Determine CUDA architecture
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Create shared library
add_library(moonai-ffi SHARED ${FFI_SOURCES} ${MOONAI_SOURCES})

# Set CUDA properties
set_target_properties(moonai-ffi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(moonai-ffi PRIVATE
    CUDA::cudart
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# Output directory
set_target_properties(moonai-ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Install rules
install(TARGETS moonai-ffi
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)