cmake_minimum_required(VERSION 3.0)

project(conveyorpp_tests)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(test_basic_primitives test_basic_primitives.cpp)
set_property(TARGET test_basic_primitives PROPERTY CXX_STANDARD 14)
target_include_directories(test_basic_primitives PRIVATE ${PROJECT_SOURCE_DIR}../include)
target_link_libraries(test_basic_primitives pthread ${GTEST_BOTH_LIBRARIES})
target_compile_options(test_basic_primitives PUBLIC ${GTEST_CFLAGS})

