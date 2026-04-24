add_library(moonai_app STATIC app.cpp)

target_include_directories(moonai_app PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_app PUBLIC
    moonai_build_config
    moonai_core
    moonai_metrics
    moonai_simulation
    moonai_evolution
    moonai_visualization
)
