#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "temporalConvolution.h"

__global__ void convolveTemporalNaiveKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int width, int height, int bufferSize, int currentIndex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float sum = 0.0f;

    for (int i = 0; i < bufferSize; i++) {
        // Compute correct index in circular buffer
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        sum += d_frames[frameIdx * width * height + idx] * d_kernel[i];
    }

    d_output[idx] = sum;
}

Mat TemporalConvolution::launchConvolveNaiveKernel(const Mat& frame, const Kernel& kernel) {


    if (frame.empty()) return frame;

    Mat converted = convert(frame);
    size_t frameSize = converted.rows * converted.cols;  // Number of float elements
    size_t frameBytes = frameSize * sizeof(float);

    float* floatFrame = new float[frameSize];
    memcpy(floatFrame, converted.ptr<float>(), frameBytes);

    gpuBuffer.enqueue(floatFrame, frameSize);

    if (!gpuBuffer.isFull()) {
        delete[] floatFrame;  // Clean up temporary buffer
        return frame;
    }

    // Allocate GPU memory only once
    static int frameIndex = 0;  // Circular buffer index
    static bool isFirstPass = true;

    cudaError_t error;
    if (isFirstPass) {
        cudaMalloc(&d_frameData, frameBytes);

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMalloc(&d_frameData...); : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        cudaMemset(d_frameData, 0, frameBytes);

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMalloc(&d_frameData...); : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        cudaMalloc(&d_convolved, frameBytes);

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMalloc(&d_convolved...); : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        cudaMemset(d_convolved, 0, frameBytes);

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMalloc(&d_kernel...); : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        cudaMalloc(&d_kernel, kernel.getSize() * sizeof(float));

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMalloc(&d_kernel...); : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        // Copy kernel data to device
        cudaMemcpy(d_kernel, kernel.getRawData(), kernel.getSize() * sizeof(float), cudaMemcpyHostToDevice);

        // For troubleshooting ------------------------------------------------------
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error at cudaMemcpy(d_kernel...) : " << cudaGetErrorString(error) << std::endl;
        }
        // ---------------------------------------------------------------------------

        isFirstPass = false;
    }

    // Compute the offset where the new frame should go
    int offset = frameIndex * frameSize;

    // For troubleshooting ------------------------------------------------------
    cout << "Copying frame to GPU. Size: " << frameBytes << " bytes" << std::endl;
    cout << "Source pointer: " << floatFrame << ", Destination offset: " << offset << std::endl;
    cout << "d_frameData + offset: " << d_frameData + offset << endl;
    std::cout << "floatFrame contents: ";

    bool hasNaN = false, hasInf = false;
    for (size_t i = 0; i < frameSize; ++i) {
        if (isnan(floatFrame[i])) hasNaN = true;
        if (isinf(floatFrame[i])) hasInf = true;
    }
    if (hasNaN) cerr << "Warning: floatFrame contains NaN values!" << endl;
    if (hasInf) cerr << "Warning: floatFrame contains Inf values!" << endl;
    // --------------------------------------------------------------------------

    // Copy the new frame into the correct position in circular buffer
    cudaMemcpy(d_frameData + offset, floatFrame, frameBytes, cudaMemcpyHostToDevice);

    // For troubleshooting ------------------------------------------------------
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error  at cudaMemcpy(d_frameData...HostToDevice) : " << cudaGetErrorString(error) << std::endl;
    }
    // ---------------------------------------------------------------------------

    cudaDeviceSynchronize();

    // Increment the frame index circularly
    frameIndex = (frameIndex + 1) % gpuBuffer.getSize();

    // Launch kernel
    int totalPixels = frame.cols * frame.rows;
    dim3 blockDim(256);
    dim3 gridDim((totalPixels + blockDim.x - 1) / blockDim.x);

    convolveTemporalNaiveKernel << <gridDim, blockDim >> > (
        d_frameData, d_convolved, d_kernel,
        frame.cols, frame.rows, gpuBuffer.getSize(), frameIndex
        );

    // For troubleshooting ------------------------------------------------------
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at kernel: " << cudaGetErrorString(error) << std::endl;
    }

    if (d_convolved == nullptr) {
        std::cerr << "Error: d_convolved is nullptr!" << std::endl;
        return frame;
    }
    // ---------------------------------------------------------------------------

    // Download result
    Mat result(converted.size(), CV_32F);
    cudaMemcpy(result.ptr<float>(), d_convolved, frameSize, cudaMemcpyDeviceToHost);

    // For troubleshooting ------------------------------------------------------
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at cudaMemcpy(result.ptr<float>()...DeviceToHost) : " << cudaGetErrorString(error) << std::endl;
    }
    // ---------------------------------------------------------------------------

    // Cleanup
    delete[] floatFrame;

    return result;
}