add_library(moonai_visualization STATIC
    visualization_manager.cpp
    renderer.cpp
    overlay.cpp
    frame_snapshot.cpp
)

target_include_directories(moonai_visualization PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_visualization PUBLIC
    moonai_build_config
    moonai_core
    moonai_evolution
    moonai_simulation
    SFML::Graphics
    SFML::Window
    SFML::System
)
