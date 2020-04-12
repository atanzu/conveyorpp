cmake_minimum_required(VERSION 3.0)

cmake_policy(SET CMP0048 NEW)
project(conveyorpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig)
pkg_search_module(GTEST REQUIRED gtest_main)

add_executable(cnvtest test/testcnv.cpp)
set_property(TARGET cnvtest PROPERTY CXX_STANDARD 14)
target_include_directories(cnvtest PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cnvtest ${GTEST_LDFLAGS})
target_compile_options(cnvtest PUBLIC ${GTEST_CFLAGS})

include(CTest)
add_test(first_and_only_test cnvtest)

# additional target to perform clang-format run, requires clang-format
file(GLOB_RECURSE ALL_SOURCE_FILES test/*.cpp include/*.h)
add_custom_target(clangformat
    COMMAND
        /usr/bin/clang-format
        -style=Google
        -i
        ${ALL_SOURCE_FILES}
)

