#ifndef CUDA_PROFILER_H
#define CUDA_PROFILER_H

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>

// Using Resource Allocation Is Initialization (RAII) design pattern
class CudaProfiler {
private:
    cudaEvent_t startEvent, stopEvent;
    // Create a cudaStream_t stream on a per class basis to avoid using
    // default stream and avoid unnecessary synchronization issues.
    // Allows for easier multithreading
    cudaStream_t stream;
    std::chrono::high_resolution_clock Clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    bool isTiming;
    bool isCPUTiming;

public:
    /**
     * @brief Initialize CUDA profiler, streams and events
     *
     */
    CudaProfiler();

    /**
     * @brief Cleans up CUDA profiler, streams and events
     *
     */
    ~CudaProfiler();

    /**
     * @brief Starts CUDA timer
     *
     * Updates internal timing state
     *
     * @return Timer start sucess
     */
    bool startTimer();

    /**
     * @brief Stops CUDA timer
     *
     * Updates internal timing state
     *
     * @return Time elapsed since startTimer() was called in milliseconds.
               -1 if error.
     */
    float stopTimer();

    /**
     * @brief Starts CPU timer
     *
     * Updates internal CPU timing state
     *
     * @return Timer start sucess
     */
    bool startCPUTimer();

    /**
     * @brief Stops CPU timer
     *
     * Updates internal CPU timing state
     *
     * @return Time elapsed since startTimer() was called in milliseconds.
               -1 if error.
     */
    float stopCPUTimer();

    /**
     * @brief Wrapper of cudaMemcpyAsync that returns time elapsed from host to 
              device
     *
     * @return Time elapsed copying from host to device
     */
    float profileMemcpyHtoD(void* d_dst, const void* h_src, size_t size);

    /**
     * @brief Wrapper of cudaMemcpyAsync that returns time elapsed from device 
              to host
     *
     * @return Time elapsed copying from device to host
     */
    float profileMemcpyDtoH(void* h_dst, const void* d_src, size_t size);

    /**
     * @brief Wrapper of cudaMemcpyAsync that returns time elapsed from device 
     *        to device
     *
     * @return Time elapsed copying from device to device
     */
    float profileMemcpyDtoD(void* d_dst, const void* d_src, size_t size);
};

#endif // CUDA_PROFILER_H