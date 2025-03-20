#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "temporalConvolution.h"

__global__ void temporalConvolveNaiveKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int frameSize, int bufferSize, int currentIndex, int tileSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frameSize) return;

    float sum = 0.0f;

    for (int i = 0; i < bufferSize; i++) {
        // Compute correct index in circular buffer
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        sum += d_frames[frameIdx * frameSize + idx] * d_kernel[i];
    }

    d_output[idx] = sum;
}

__global__ void temporalConvolveSharedMemKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int frameSize, int bufferSize, int currentIndex, int tileSize)
{
    // Allocate shared memory for both kernel and frame data samples
    extern __shared__ float sharedMem[];

    // Split shared memory: first part for kernel, second part for frame samples
    float* s_kernel = sharedMem;
    float* s_frames = &sharedMem[bufferSize];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frameSize) return;

    // Cooperatively load kernel into shared memory
    if (threadIdx.x < bufferSize) {
        s_kernel[threadIdx.x] = d_kernel[threadIdx.x];
    }

    // Each thread loads its own samples from the different frames
    for (int i = 0; i < bufferSize; i++) {
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        s_frames[threadIdx.x * bufferSize + i] = d_frames[frameIdx * frameSize + idx];
    }

    __syncthreads(); // Ensure all data is loaded

    // Compute convolution using shared memory
    float sum = 0.0f;
    for (int i = 0; i < bufferSize; i++) {
        sum += s_frames[threadIdx.x * bufferSize + i] * s_kernel[i];
    }

    d_output[idx] = sum;
}

TemporalConvolveKernelPtr getTemporalConvolveNaiveKernel() {
    return temporalConvolveNaiveKernel;
}

TemporalConvolveKernelPtr getTemporalConvolveSharedMemKernel() {
    return temporalConvolveSharedMemKernel;
}

Mat TemporalConvolution::launchConvolveKernel(const Mat& frame, const Kernel& kernel, 
    int blockSize, const int tileSize, const bool useSharedMemory,
    void (*convolveKernel)(float*, float*, float*, int, int, int, int)) {

    if (frame.empty()) return frame;

    Mat converted = convert(frame);
    size_t frameSize = converted.rows * converted.cols;  // Number of float elements
    size_t frameBytes = frameSize * sizeof(float);

    float* floatFrame = new float[frameSize];
    memcpy(floatFrame, converted.ptr<float>(), frameBytes);

    gpuBuffer.enqueue(floatFrame, frameSize);

    if (!gpuBuffer.isFull()) {
        delete[] floatFrame;  // Clean up temporary buffer
        return Mat();
    }

    if (isFirstPass) {
        cudaMalloc(&d_frames, frameBytes * kernel.getSize());
        cudaMemset(d_frames, 0, frameBytes * kernel.getSize());

        cudaMalloc(&d_convolved, frameBytes);
        cudaMemset(d_convolved, 0, frameBytes);

        cudaMalloc(&d_kernel, kernel.getSize() * sizeof(float));
        cudaMemcpy(d_kernel, kernel.getRawData(), kernel.getSize() * sizeof(float), cudaMemcpyHostToDevice);

        isFirstPass = false;
    }

    // Compute the offset where the new frame should go
    int offset = frameIndex * frameSize;

    // Copy the new frame into the correct position in circular buffer
    cudaMemcpy(d_frames + offset, floatFrame, frameBytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Increment the frame index circularly
    frameIndex = (frameIndex + 1) % gpuBuffer.getSize();

    if (blockSize == 0) {
        blockSize = 256;
    } 

    int sharedMemSize = 0;
    if (useSharedMemory) {
        sharedMemSize = kernel.getSize() * sizeof(float) +
            blockSize * kernel.getSize() * sizeof(float);
    }

    convolveKernel << <(frameSize + blockSize - 1) / blockSize, blockSize, sharedMemSize >> > (
        d_frames, d_convolved, d_kernel, frameSize, gpuBuffer.getSize(), frameIndex, tileSize);

    // Download result
    Mat result(converted.size(), CV_32F);
    cudaMemcpy(result.ptr<float>(), d_convolved, frameBytes, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] floatFrame;

    return result;
}