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

# Main Project setup
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Source and header files
set(MAIN_SOURCES
    OpticalFlow/circularBuffer.cpp
    OpticalFlow/convolution.cpp
    OpticalFlow/kernel.cpp
	OpticalFlow/least_squares_solver.cpp
	OpticalFlow/least_squares_solver_cuda.cu
    OpticalFlow/profiler.cu
    OpticalFlow/spatialConvolution.cpp
    OpticalFlow/spatialConvolution.cu
    OpticalFlow/temporalConvolution.cpp
    OpticalFlow/temporalConvolution.cu
	OpticalFlow/utilities.cpp
)

set(MAIN_HEADERS
    OpticalFlow/circularBuffer.h
    OpticalFlow/collection_adapters.hpp
    OpticalFlow/convolution.h
    OpticalFlow/kernel.h
	OpticalFlow/least_squares_solver.h
	OpticalFlow/least_squares_solver_cuda.h
    OpticalFlow/profiler.h
    OpticalFlow/spatialConvolution.h
    OpticalFlow/temporalConvolution.h
	OpticalFlow/utilities.h
)

set(CUDA_SOURCES
    OpticalFlow/test.cu
    OpticalFlow/profiler.cu
)

set(CUDA_HEADERS
    OpticalFlow/profiler.h
)

# Arrow library
find_package(Arrow REQUIRED)
# Download the rerun_sdk
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN_INCLUDE_DIR})

### OPENCV Combination Driver Project
add_executable(driver OpticalFlow/driver.cu ${MAIN_SOURCES} ${MAIN_HEADERS})
set_target_properties(driver PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_compile_options(driver PRIVATE -Xcompiler "/EHsc /Zi /Ob0")
target_link_libraries(driver ${OpenCV_LIBS} rerun_sdk Eigen3::Eigen)

### OPENCV CPU Driver Project
add_executable(driverCPU OpticalFlow/driverCPU.cu ${MAIN_SOURCES} ${MAIN_HEADERS})
set_target_properties(driverCPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_compile_options(driverCPU PRIVATE -Xcompiler "/EHsc /Zi /Ob0")
target_link_libraries(driverCPU ${OpenCV_LIBS} rerun_sdk Eigen3::Eigen)

### OPENCV GPU Driver Project
add_executable(driverGPU OpticalFlow/driverGPU.cu ${MAIN_SOURCES} ${MAIN_HEADERS})
set_target_properties(driverGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_compile_options(driverGPU PRIVATE -Xcompiler "/EHsc /Zi /Ob0")
target_link_libraries(driverGPU ${OpenCV_LIBS} rerun_sdk Eigen3::Eigen)

### OPENCV GPU Driver Project
add_executable(driverGPUSharedMem OpticalFlow/driverGPUSharedMem.cu ${MAIN_SOURCES} ${MAIN_HEADERS})
set_target_properties(driverGPUSharedMem PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_compile_options(driverGPUSharedMem PRIVATE -Xcompiler "/EHsc /Zi /Ob0")
target_link_libraries(driverGPUSharedMem ${OpenCV_LIBS} rerun_sdk Eigen3::Eigen)

### Test CUDA Project
add_executable(test_cuda ${CUDA_SOURCES} ${CUDA_HEADERS})
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON EXCLUDE_FROM_ALL ON)  # Exclude from ALL_BUILD
target_link_libraries(test_cuda PRIVATE rerun_sdk)

#### Example_minimal Project
add_executable(example_minimal main.cpp)
target_link_libraries(example_minimal PRIVATE rerun_sdk)

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

foreach(DLL ${OPENCV_DLLS})
    add_custom_command(TARGET driverCPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" $<TARGET_FILE_DIR:driverCPU>
    )
endforeach()

foreach(DLL ${OPENCV_DLLS})
    add_custom_command(TARGET driverGPU POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" $<TARGET_FILE_DIR:driverGPU>
    )
endforeach()

foreach(DLL ${OPENCV_DLLS})
    add_custom_command(TARGET driverGPUSharedMem POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" $<TARGET_FILE_DIR:driverGPUSharedMem>
    )
endforeach()

foreach(DLL ${OPENCV_DLLS})
    add_custom_command(TARGET example_minimal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" $<TARGET_FILE_DIR:example_minimal>
    )
endforeach()
