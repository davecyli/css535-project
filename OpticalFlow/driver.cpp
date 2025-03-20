/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/

#include "spatialConvolution.h"
#include "temporalConvolution.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // Check input validity
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video file path>" << std::endl;
        return 1;
    }

    string video = argv[1];
    VideoCapture capture(video);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open video file " << video << endl;
        return -1;
    }

    // Create a 1D Gaussian kernel
    int kernelSizeSmoothing = 25;
    float sigma = 3.2f;
    Kernel* gaussianKernel = Kernel::generateGaussian(kernelSizeSmoothing, sigma);

    // Create a 1D Derivative kernel
    int kernelSizeDerivative = 5;
    float derivativeKernelArray[] = { 1.0f / 12, -8.0f / 12, 0.0f, 8.0f / 12, -1.0f / 12 };
    Kernel derivativeKernel(derivativeKernelArray, kernelSizeDerivative);

    // Process frames
    SpatialConvolution cpuXY(Implementation::CPU_NAIVE);
    TemporalConvolution cpuSmoothT(Implementation::CPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution cpuDerivativeT(Implementation::CPU_NAIVE, kernelSizeDerivative);
    Mat frame, cpuSmoothedT, cpuSmoothedTX, cpuSmoothedTXY, cpuI_t, cpuI_x, cpuI_y, derivativeDisplay;

    SpatialConvolution gpuXY(Implementation::GPU_NAIVE);
    TemporalConvolution gpuSmoothT(Implementation::GPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution gpuDerivativeT(Implementation::GPU_NAIVE, kernelSizeDerivative);
    Mat gpuSmoothedT, gpuSmoothedTX, gpuSmoothedTXY, gpuI_t, gpuI_x, gpuI_y;

    while (capture.read(frame)) {
        // Display original image
        imshow("Original image", frame);
        waitKey(1);

        // CPU Implementation ---------------------------------------------------------------------

        cpuSmoothedT = cpuSmoothT.convolve(frame, *gaussianKernel);
        if (!cpuSmoothedT.empty()) {
            imshow("Smoothed T: CPU", cpuSmoothedT);
            waitKey(1);
        }

        cpuSmoothedTX = cpuXY.convolveX(cpuSmoothedT, *gaussianKernel);
        if (!cpuSmoothedTX.empty()) {
            imshow("Smoothed TX: CPU", cpuSmoothedTX);
            waitKey(1);
        }

        cpuSmoothedTXY = cpuXY.convolveY(cpuSmoothedTX, *gaussianKernel);
        if (!cpuSmoothedTXY.empty()) {
            imshow("Smoothed TXY: CPU", cpuSmoothedTXY);
            waitKey(1);
        }
        
        cpuI_t = cpuDerivativeT.convolve(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_t.empty()) {
            normalize(cpuI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_t: CPU", derivativeDisplay);
            waitKey(1);
        }

        cpuI_x = cpuXY.convolveX(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_x.empty()) {
            normalize(cpuI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_x: CPU", derivativeDisplay);
            waitKey(1);
        }

        cpuI_y = cpuXY.convolveY(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_y.empty()) {
            normalize(cpuI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_y: CPU", derivativeDisplay);
            waitKey(1);
        }

        // GPU Implementation ---------------------------------------------------------------------

        gpuSmoothedT = gpuSmoothT.convolve(frame, *gaussianKernel);
        if (!gpuSmoothedT.empty()) {
            imshow("Smoothed T: GPU Naive", gpuSmoothedT);
            waitKey(1);
        }

        gpuSmoothedTX = gpuXY.convolveX(gpuSmoothedT, *gaussianKernel);
        if (!gpuSmoothedTX.empty()) {
            imshow("Smoothed TX: GPU Naive", gpuSmoothedTX);
            waitKey(1);
        }

        gpuSmoothedTXY = gpuXY.convolveY(gpuSmoothedTX, *gaussianKernel);
        if (!gpuSmoothedTXY.empty()) {
            imshow("Smoothed TXY: GPU Naive", gpuSmoothedTXY);
            waitKey(1);
        }

        gpuI_t = gpuDerivativeT.convolve(gpuSmoothedTXY, derivativeKernel);
        if (!gpuI_t.empty()) {
            normalize(gpuI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("gpuI_t: GPU Naive", derivativeDisplay);
            waitKey(1);
        }

        gpuI_x = gpuXY.convolveX(gpuSmoothedTXY, derivativeKernel);
        if (!gpuI_x.empty()) {
            normalize(gpuI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("gpuI_x: GPU Naive", derivativeDisplay);
            waitKey(1);
        }

        gpuI_y = gpuXY.convolveY(gpuSmoothedTXY, derivativeKernel);
        if (!gpuI_y.empty()) {
            normalize(gpuI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("gpuI_y: GPU Naive", derivativeDisplay);
            waitKey(1);
        }

    }

    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    destroyAllWindows();

    return 0;
}