/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/
#include "profiler.h"
#include "spatialConvolution.h"
#include "temporalConvolution.h"
#include "collection_adapters.hpp"

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
    CudaProfiler profiler = CudaProfiler();
    string videoFilePath = "";
    // Check input validity
    if (argc < 2) {
        std::cout << "Enter video file path (..\\OpticalFlow\\data\\Squishies.mp4 if none entered): ";
        std::getline(std::cin, videoFilePath);
        if (videoFilePath == "") {
            videoFilePath = "..\\OpticalFlow\\data\\Squishies.mp4";
        }
    } else {
        videoFilePath = argv[1];
    }

    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const rerun::RecordingStream rec = rerun::RecordingStream("OpticalFlow");
    // Try to spawn a new viewer instance.
    rerun::Error error = rec.spawn();

    VideoCapture capture(videoFilePath);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open video file " << videoFilePath << endl;
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
    int index = 0;

    // Process frames
    SpatialConvolution spatial;
    TemporalConvolution temporalSmooth(kernelSizeSmoothing);
    TemporalConvolution temporalDerivative(kernelSizeDerivative);
    Mat frame, smoothedT, smoothedTX, smoothedTXY, I_t, I_x, I_y, derivativeDisplay, smoothedDisplay;
    cv::Mat frameRGBA;
    while (capture.read(frame)) {
        // Display original image
        // Convert from BGR to RGBA
        cv::cvtColor(frame, frameRGBA, cv::COLOR_BGR2RGBA);
        // Log image to rerun using the tensor buffer adapter defined in `collection_adapters.hpp`.
        rec.log("0.input", rerun::Image::from_rgba32(frameRGBA, { uint32_t(frameRGBA.cols), uint32_t(frameRGBA.rows) }));

        profiler.startCPUTimer();
        smoothedT = temporalSmooth.convolve(frame, *gaussianKernel);
        if (!smoothedT.empty()) {
            rec.log("1.T.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(smoothedT, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("1.T", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        smoothedTX = spatial.convolveX(smoothedT, *gaussianKernel);
        if (!smoothedTX.empty()) {
            rec.log("2.TX.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(smoothedTX, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("2.TX", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        smoothedTXY = spatial.convolveY(smoothedTX, *gaussianKernel);
        if (!smoothedTXY.empty()) {
            rec.log("3.TXY.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(smoothedTXY, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("3.TXY", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        I_t = temporalDerivative.convolve(smoothedTXY, derivativeKernel);
        if (!I_t.empty()) {
            rec.log("4.I_t.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(I_t, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("4.I_t", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        I_x = spatial.convolveX(smoothedTXY, derivativeKernel);
        if (!I_x.empty()) {
            rec.log("5.I_x.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(I_x, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("5.I_x", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        I_y = spatial.convolveY(smoothedTXY, derivativeKernel);
        if (!I_y.empty()) {
            rec.log("6.I_y.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(I_y, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("6.I_y", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

    }

    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    //destroyAllWindows();

    return 0;
}