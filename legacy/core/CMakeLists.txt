set(MOONAI_CORE_SOURCES
    app_state.cpp
    config.cpp
    lua_runtime.cpp
    random.cpp
)

add_library(moonai_core STATIC ${MOONAI_CORE_SOURCES})

target_include_directories(moonai_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_core PUBLIC
    moonai_build_config
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    sol2::sol2
    ${LUA_LIBRARIES}
)

target_include_directories(moonai_core PUBLIC ${LUA_INCLUDE_DIR})
