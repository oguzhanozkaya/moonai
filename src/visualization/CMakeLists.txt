add_library(moonai_visualization STATIC
    visualization_manager.cpp
    renderer.cpp
    overlay.cpp
    system_font_resolver.cpp
    frame_snapshot.cpp
)

target_include_directories(moonai_visualization PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_visualization PUBLIC
    moonai_core
    moonai_evolution
    moonai_simulation
    SFML::Graphics
    SFML::Window
    SFML::System
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(moonai_visualization PUBLIC Fontconfig::Fontconfig)
endif()

if(WIN32)
    target_link_libraries(moonai_visualization PUBLIC dwrite)
endif()

if(MOONAI_BUILD_PROFILER)
    add_library(moonai_visualization_profiler STATIC
        visualization_manager.cpp
        renderer.cpp
        overlay.cpp
        system_font_resolver.cpp
        frame_snapshot.cpp
    )
    target_include_directories(moonai_visualization_profiler PUBLIC ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moonai_visualization_profiler PUBLIC
        moonai_core_profiler
        moonai_evolution_profiler
        moonai_simulation_profiler
        SFML::Graphics
        SFML::Window
        SFML::System
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(moonai_visualization_profiler PUBLIC Fontconfig::Fontconfig)
    endif()

    if(WIN32)
        target_link_libraries(moonai_visualization_profiler PUBLIC dwrite)
    endif()

    target_compile_definitions(moonai_visualization_profiler PUBLIC MOONAI_BUILD_PROFILER)
endif()
