cmake_minimum_required(VERSION 3.0)

cmake_policy(SET CMP0048 NEW)
project(conveyorpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
    EXPORT ${PROJECT_NAME}Targets 
    DESTINATION lib/cmake/${PROJECT_NAME}
)
install(
    FILES 
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION
        lib/cmake/${PROJECT_NAME}
)
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include/${PROJECT_NAME}
)


find_package(conveyorpp REQUIRED)
find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest_main)

add_executable(test_basic_primitives test/test_basic_primitives.cpp)
set_property(TARGET test_basic_primitives PROPERTY CXX_STANDARD 14)
target_link_libraries(test_basic_primitives conveyorpp ${GTEST_LDFLAGS})
target_compile_options(test_basic_primitives PUBLIC ${GTEST_CFLAGS})

include(CTest)
add_test(basic_primitives test_basic_primitives)


# additional target to perform clang-format run, requires clang-format
file(GLOB_RECURSE ALL_SOURCE_FILES test/*.cpp include/*.h)
add_custom_target(clangformat
    COMMAND
        /usr/bin/clang-format
        -style=Google
        -i
        ${ALL_SOURCE_FILES}
)

