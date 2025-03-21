/*

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

    }

    capture.release();
    capture.open(video); 

    SpatialConvolution gpuNaiveXY(Implementation::GPU_NAIVE);
    TemporalConvolution gpuNaiveSmoothT(Implementation::GPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution gpuNaiveDerivativeT(Implementation::GPU_NAIVE, kernelSizeDerivative);
    Mat gpuNaiveSmoothedT, gpuNaiveSmoothedTX, gpuNaiveSmoothedTXY, gpuNaiveI_t, gpuNaiveI_x, gpuNaiveI_y;

    int blockSizes[] = { 1 << 10, 1 << 9, 1 << 8, 1 << 7, 1 << 6, 1 << 5 };
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    for (int i = 0; i < numBlockSizes; i++) {
        while (capture.read(frame)) {

            // GPU Implementation ---------------------------------------------------------------------

            int blockSize = blockSizes[i];

            gpuNaiveSmoothedT = gpuNaiveSmoothT.convolve(frame, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedT.empty()) {
                string windowName = "Smoothed T, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedT);
                waitKey(1);
            }

            gpuNaiveSmoothedTX = gpuNaiveXY.convolveX(gpuNaiveSmoothedT, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedTX.empty()) {
                string windowName = "Smoothed TX, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedTX);
                waitKey(1);
            }

            gpuNaiveSmoothedTXY = gpuNaiveXY.convolveY(gpuNaiveSmoothedTX, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedTXY.empty()) {
                string windowName = "Smoothed TXY, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedTXY);
                waitKey(1);
            }

            gpuNaiveI_t = gpuNaiveDerivativeT.convolve(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_t.empty()) {
                normalize(gpuNaiveI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_t, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }


            gpuNaiveI_x = gpuNaiveXY.convolveX(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_x.empty()) {
                normalize(gpuNaiveI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_x, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }


            gpuNaiveI_y = gpuNaiveXY.convolveY(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_y.empty()) {
                normalize(gpuNaiveI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_y, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
        }
        capture.release();
        capture.open(video);
    }

    SpatialConvolution gpuSharedMemXY(Implementation::GPU_SHARED_MEMORY);
    TemporalConvolution gpuSharedMemSmoothT(Implementation::GPU_SHARED_MEMORY, 
        kernelSizeSmoothing);
    TemporalConvolution gpuSharedMemDerivativeT(Implementation::GPU_SHARED_MEMORY, 
        kernelSizeDerivative);
    Mat gpuSharedMemSmoothedT, gpuSharedMemSmoothedTX, gpuSharedMemSmoothedTXY, 
        gpuSharedMemI_t, gpuSharedMemI_x, gpuSharedMemI_y;

    for (int i = 0; i < numBlockSizes; i++) {
        while (capture.read(frame)) {

            // GPU Implementation ---------------------------------------------------------------------

            int blockSize = blockSizes[i];

            int maxBlockSizeT = 1 << 8;
            if (blockSize > maxBlockSizeT) {
                gpuSharedMemSmoothedT = gpuSharedMemSmoothT.convolve(
                    frame, *gaussianKernel, maxBlockSizeT);
                if (!gpuSharedMemSmoothedT.empty()) {
                    string windowName = "Smoothed T, GPU Shared Memory, Block Size: "
                        + to_string(maxBlockSizeT);
                    imshow(windowName, gpuSharedMemSmoothedT);
                    waitKey(1);
                }
            } else {
                gpuSharedMemSmoothedT = gpuSharedMemSmoothT.convolve(
                    frame, *gaussianKernel, blockSize);
                if (!gpuSharedMemSmoothedT.empty()) {
                    string windowName = "Smoothed T, GPU Shared Memory, Block Size: "
                        + to_string(blockSize);
                    imshow(windowName, gpuSharedMemSmoothedT);
                    waitKey(1);
                }
            }

            gpuSharedMemSmoothedTX = gpuSharedMemXY.convolveX(
                gpuSharedMemSmoothedT, *gaussianKernel, blockSize);
            if (!gpuSharedMemSmoothedTX.empty()) {
                string windowName = "Smoothed TX, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, gpuSharedMemSmoothedTX);
                waitKey(1);
            }
            
            gpuSharedMemSmoothedTXY = gpuSharedMemXY.convolveY(
                gpuSharedMemSmoothedTX, *gaussianKernel, blockSize);
            if (!gpuSharedMemSmoothedTXY.empty()) {
                string windowName = "Smoothed TXY, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, gpuSharedMemSmoothedTXY);
                waitKey(1);
            }
            
            gpuSharedMemI_t = gpuSharedMemDerivativeT.convolve(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_t.empty()) {
                normalize(gpuSharedMemI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_t, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
            
            gpuSharedMemI_x = gpuSharedMemXY.convolveX(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_x.empty()) {
                normalize(gpuSharedMemI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_x, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }

            gpuSharedMemI_y = gpuSharedMemXY.convolveY(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_y.empty()) {
                normalize(gpuSharedMemI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_y, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
            
        }
    capture.release();
    capture.open(video);
    }
    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    destroyAllWindows();

    return 0;
}