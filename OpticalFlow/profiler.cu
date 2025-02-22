#include "profiler.h"

CudaProfiler::CudaProfiler() : isTiming(false) {
    cudaProfilerStart();
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaStreamCreate(&stream);
}

CudaProfiler::~CudaProfiler() {
    cudaProfilerStop();
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);
}

bool CudaProfiler::startTimer() {
    if (isTiming) {
        std::cerr << "Error: A timer is already running!\n";
        return false;
    }
    cudaEventRecord(startEvent, stream);
    isTiming = true;
    return true;
}

float CudaProfiler::stopTimer() {
    if (!isTiming) {
        std::cerr << "Error: Timer was not started!\n";
        return -1.0f;
    }
    cudaEventRecord(stopEvent, stream);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    isTiming = false;
    return milliseconds;
}

float CudaProfiler::profileMemcpyHtoD(void* d_dst, const void* h_src, size_t size) {
    if (!startTimer()) return -1.0f;
    //cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    return stopTimer();
}

float CudaProfiler::profileMemcpyDtoH(void* h_dst, const void* d_src, size_t size) {
    if (!startTimer()) return -1.0f;
    //cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    return stopTimer();
}

float CudaProfiler::profileMemcpyDtoD(void* d_dst, const void* d_src, size_t size) {
    if (!startTimer()) return -1.0f;
    //cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
    return stopTimer();
}