#include "least_squares_solver_cuda.h"

#define WINDOW_SIZE 5
#define HALF_WIN (WINDOW_SIZE / 2)
#define MIN_DETERMINANT 0

// Constructor
LeastSquaresSolverCUDA::LeastSquaresSolverCUDA() : d_Ix(nullptr), d_Iy(nullptr), d_It(nullptr), d_flowX(nullptr), d_flowY(nullptr) {}

// Destructor
LeastSquaresSolverCUDA::~LeastSquaresSolverCUDA() {
    freeMemory();
}

// GPU Kernel: Computes optical flow using a weighted least squares method
__global__ void computeOpticalFlowKernel(
    float* d_Ix, float* d_Iy, float* d_It, 
    float* d_flowX, float* d_flowY, 
    int width, int height) {

    // Compute pixel index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < HALF_WIN || x >= width - HALF_WIN || y < HALF_WIN || y >= height - HALF_WIN)
        return; // Skip boundary pixels

    // Define weights
    const float weights[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };

    // A = 2x2 matrix, B = 2x1 vector
    float A[2][2] = { {0, 0}, {0, 0} };
    float B[2] = { 0, 0 };

    // Compute weighted A and B
    for (int i = -HALF_WIN; i <= HALF_WIN; i++) {
        for (int j = -HALF_WIN; j <= HALF_WIN; j++) {
            float w = weights[HALF_WIN + i] * weights[HALF_WIN + j]; // Compute 2D weight

            int idx = (y + i) * width + (x + j);
            float ix = d_Ix[idx];
            float iy = d_Iy[idx];
            float it = d_It[idx];

            A[0][0] += w * w * (ix * ix);
            A[0][1] += w * w * (ix * iy);
            A[1][0] += w * w * (ix * iy);
            A[1][1] += w * w * (iy * iy);

            B[0] -= w * w * (ix * it);
            B[1] -= w * w * (iy * it);
        }
    }

    // Compute determinant
    float detA = A[0][0] * A[1][1] - A[0][1] * A[1][0];

    if (detA > MIN_DETERMINANT) {
        // Compute inverse of A
        float invA[2][2];
        invA[0][0] = A[1][1] / detA;
        invA[0][1] = -A[0][1] / detA;
        invA[1][0] = -A[1][0] / detA;
        invA[1][1] = A[0][0] / detA;

        // Solve for (Vx, Vy)
        float Vx = invA[0][0] * B[0] + invA[0][1] * B[1];
        float Vy = invA[1][0] * B[0] + invA[1][1] * B[1];

        // Store results
        int out_idx = y * width + x;
        d_flowX[out_idx] = Vx;
        d_flowY[out_idx] = Vy;
    }
}

// Memory allocation
void LeastSquaresSolverCUDA::allocateMemory(int w, int h) {
    if (d_Ix) freeMemory(); // Free old memory if re-allocating

    width = w;
    height = h;
    int size = width * height * sizeof(float);

    cudaMalloc(&d_Ix, size);
    cudaMalloc(&d_Iy, size);
    cudaMalloc(&d_It, size);
    cudaMalloc(&d_flowX, size);
    cudaMalloc(&d_flowY, size);
}

// Free GPU memory
void LeastSquaresSolverCUDA::freeMemory() {
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_It);
    cudaFree(d_flowX);
    cudaFree(d_flowY);
}

// Compute optical flow
void LeastSquaresSolverCUDA::computeOpticalFlow(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It, cv::Mat& flowX, cv::Mat& flowY) {
    int w = Ix.cols;
    int h = Ix.rows;
    int size = w * h * sizeof(float);

    allocateMemory(w, h);

    // Copy data to GPU
    cudaMemcpy(d_Ix, Ix.ptr<float>(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Iy, Iy.ptr<float>(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_It, It.ptr<float>(), size, cudaMemcpyHostToDevice);

    // Define CUDA execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    computeOpticalFlowKernel<<<gridDim, blockDim>>>(d_Ix, d_Iy, d_It, d_flowX, d_flowY, w, h);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(flowX.ptr<float>(), d_flowX, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(flowY.ptr<float>(), d_flowY, size, cudaMemcpyDeviceToHost);
}
