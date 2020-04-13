cmake_minimum_required(VERSION 3.0)

project(conveyorpp_tests)

enable_testing()
find_package(Threads)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(test_basic_primitives test_basic_primitives.cpp)
set_property(TARGET test_basic_primitives PROPERTY CXX_STANDARD 14)
target_link_libraries(test_basic_primitives
    PRIVATE
        conveyorpp
        gtest
        ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_options(test_basic_primitives PUBLIC ${GTEST_CFLAGS})

