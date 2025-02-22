/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/
#include "kernel.h"
#include "convolution.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {

    string video = R"(C:\Users\akoser\Documents\MSCSSE\CSS_535\Project\Squishies.mp4)";
    VideoCapture cap(video);
    Mat frame, smoothedXY, smoothedXYT, derivativeXY, derivativeXYT, derivativeDisplay;

    // Create a 1D Gaussian kernel
    int kernelSizeSmoothing = 25;
    float sigma = 3.2;
    Kernel* gaussianKernel = Kernel::generateGaussian(kernelSizeSmoothing, sigma);

    // Create a 1D Derivative kernel
    int kernelSizeDerivative = 5;
    float derivativeKernelArray[] = { 1.0f / 12, -8.0f / 12, 0.0f, 8.0f / 12, -1.0f / 12 };
    Kernel derivativeKernel(derivativeKernelArray, kernelSizeDerivative);

    // Print the kernel
    cout << "Gaussian Kernel:" << endl;
    gaussianKernel->print();

    Convolution convolution;

    deque<Mat> smoothedFramesXY;  // Input to temporal smoothing
    deque<Mat> derivativeFramesXY; // Input to temporal derivative
    // Loop through frames
    while (cap.read(frame)) {

        // Convert to grayscale
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // Convert to float
        frame.convertTo(frame, CV_32F, 1.0 / 255.0);
        imshow("Original image (grayscale)", frame);
        waitKey(1);

        // Spatial smoothing
        smoothedXY = convolution.convolveXY(frame, *gaussianKernel);
        imshow("Image smoothed in XY", smoothedXY);
        waitKey(1);

        // Add to temporal smoothing input buffer
        smoothedFramesXY.push_back(smoothedXY.clone());
        if (smoothedFramesXY.size() > kernelSizeSmoothing) {
            smoothedFramesXY.pop_front();
        }

        // If buffer has enough frames for temporal smoothing
        if (smoothedFramesXY.size() == kernelSizeSmoothing) {
            smoothedXYT = convolution.convolveT(*gaussianKernel, smoothedFramesXY);
            imshow("Smoothed XYT", smoothedXYT);
            waitKey(1);

            derivativeXY = convolution.convolveXY(smoothedXYT, derivativeKernel);

            // For display only
            normalize(derivativeXY, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("Derivative XY", derivativeDisplay);
            waitKey(1);

            // Add to temporal derivative input buffer
            derivativeFramesXY.push_back(derivativeXY.clone());
            if (derivativeFramesXY.size() > kernelSizeDerivative) {
                derivativeFramesXY.pop_front();
            }

            // If buffer has enough frames for temporal derivative calculation
            if (derivativeFramesXY.size() == kernelSizeDerivative) {
                derivativeXYT = convolution.convolveT(derivativeKernel, derivativeFramesXY);

                // For display only
                normalize(derivativeXY, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                imshow("Derivative XYT", derivativeDisplay);
                waitKey(1);
            }
        }  
    }

    // Cleanup
    delete gaussianKernel;
    cap.release();
    destroyAllWindows();

	return 0;
}