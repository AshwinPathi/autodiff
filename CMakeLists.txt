cmake_minimum_required(VERSION 3.15)
project(my_project VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directory for headers
include_directories(include)

# Collect all source files (or list them manually)
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(my_project ${SOURCES})

# ------------------------------
# AddressSanitizer toggle
# ------------------------------
option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(ASAN_FLAGS "-fsanitize=address")
    add_compile_options(${ASAN_FLAGS})
    add_link_options(${ASAN_FLAGS})
endif()

# ------------------------------
# GoogleTest
# ------------------------------
include(FetchContent)

# Prevent gtest from building its own tests
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

file(GLOB TEST_SOURCES "tests/*.cpp")

if(TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
        gtest_discover_tests(${test_name})
    endforeach()
endif()