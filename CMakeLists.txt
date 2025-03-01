cmake_minimum_required(VERSION 3.18)
project(OpticalFlow LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_SUPPRESS_REGENERATION ON) # Prevents ZERO_CHECK from appearing

# Suppress deprecated GPU target warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# OpenCV setup
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Source and header files
set(OCV_SOURCES
    OpticalFlow/driver.cpp
    OpticalFlow/kernel.cpp
    OpticalFlow/convolution.cpp
    OpticalFlow/spatialConvolution.cpp
    OpticalFlow/temporalConvolution.cpp
    OpticalFlow/circularBuffer.cpp
)

set(OCV_HEADERS
    OpticalFlow/spatialConvolution.h
    OpticalFlow/temporalConvolution.h
    OpticalFlow/kernel.h
    OpticalFlow/convolution.h
    OpticalFlow/circularBuffer.h
)

set(CUDA_SOURCES
    OpticalFlow/test.cu
    OpticalFlow/profiler.cu
)

# Create OpenCV-based executable
add_executable(driver ${OCV_SOURCES} ${OCV_HEADERS})
set_target_properties(driver PROPERTIES EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_compile_options(driver PRIVATE /EHsc)
target_link_libraries(driver ${OpenCV_LIBS})

# Set the default startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT driver)

# Copy necessary OpenCV DLLs after build
set(OPENCV_DLLS
    "${OpenCV_DIR}/x64/vc16/bin/opencv_world4110.dll"
    "${OpenCV_DIR}/x64/vc16/bin/opencv_world4110d.dll"
    "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_ffmpeg4110_64.dll"
    "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_msmf4110_64.dll"
    "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_msmf4110_64d.dll"
)

foreach(DLL ${OPENCV_DLLS})
    add_custom_command(TARGET driver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" $<TARGET_FILE_DIR:driver>
    )
endforeach()

# Create CUDA executable
add_executable(test_cuda ${CUDA_SOURCES})
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
