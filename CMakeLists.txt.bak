cmake_minimum_required(VERSION 3.18)
project(example_minimal LANGUAGES CXX CUDA)

add_executable(example_minimal main.cpp)

find_package(Arrow REQUIRED)

# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Link against rerun_sdk.
target_link_libraries(example_minimal PRIVATE rerun_sdk)
