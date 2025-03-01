/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/

#include "spatialConvolution.h"
#include "temporalConvolution.h"

#include <Eigen>

#include <opencv2/opencv.hpp>

#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

using namespace std;
using namespace cv;
using namespace rerun::demo;


template <>
struct rerun::CollectionAdapter<uint8_t, cv::Mat>
{
    /* Adapters to borrow an OpenCV image into Rerun
     * images without copying */

    Collection<uint8_t> operator()(const cv::Mat& img)
    {
        // Borrow for non-temporary.

        assert("OpenCV matrix expected have bit depth CV_U8" && CV_MAT_DEPTH(img.type()) == CV_8U);
        return Collection<uint8_t>::borrow(img.data, img.total() * img.channels());
    }

    Collection<uint8_t> operator()(cv::Mat&& img)
    {
        /* Do a full copy for temporaries (otherwise the data
         * might be deleted when the temporary is destroyed). */

        assert("OpenCV matrix expected have bit depth CV_U8" && CV_MAT_DEPTH(img.type()) == CV_8U);
        std::vector<uint8_t> img_vec(img.total() * img.channels());
        img_vec.assign(img.data, img.data + img.total() * img.channels());
        return Collection<uint8_t>::take_ownership(std::move(img_vec));
    }
};


template <>
struct rerun::CollectionAdapter<rerun::Position3D, std::vector<Eigen::Vector3f>>
{
    /* Adapters to log eigen vectors as rerun positions*/

    Collection<rerun::Position3D> operator()(const std::vector<Eigen::Vector3f>& container)
    {
        // Borrow for non-temporary.
        return Collection<rerun::Position3D>::borrow(container.data(), container.size());
    }

    Collection<rerun::Position3D> operator()(std::vector<Eigen::Vector3f>&& container)
    {
        /* Do a full copy for temporaries (otherwise the data
         * might be deleted when the temporary is destroyed). */
        std::vector<rerun::Position3D> positions(container.size());
        memcpy(positions.data(), container.data(), container.size() * sizeof(Eigen::Vector3f));
        return Collection<rerun::Position3D>::take_ownership(std::move(positions));
    }
};


template <>
struct rerun::CollectionAdapter<rerun::Position3D, Eigen::Matrix3Xf>
{
    /* Adapters so we can log an eigen matrix as rerun positions */

    // Sanity check that this is binary compatible.
    static_assert(
        sizeof(rerun::Position3D) == sizeof(Eigen::Matrix3Xf::Scalar) * Eigen::Matrix3Xf::RowsAtCompileTime
        );

    Collection<rerun::Position3D> operator()(const Eigen::Matrix3Xf& matrix)
    {
        // Borrow for non-temporary.
        static_assert(alignof(rerun::Position3D) <= alignof(Eigen::Matrix3Xf::Scalar));
        return Collection<rerun::Position3D>::borrow(
            // Cast to void because otherwise Rerun will try to do above sanity checks with the wrong type (scalar).
            reinterpret_cast<const void*>(matrix.data()),
            matrix.cols()
        );
    }

    Collection<rerun::Position3D> operator()(Eigen::Matrix3Xf&& matrix)
    {
        /* Do a full copy for temporaries (otherwise the
         * data might be deleted when the temporary is destroyed). */
        std::vector<rerun::Position3D> positions(matrix.cols());
        memcpy(positions.data(), matrix.data(), matrix.size() * sizeof(rerun::Position3D));
        return Collection<rerun::Position3D>::take_ownership(std::move(positions));
    }
};



int main(int argc, char* argv[]) {
    string videoFilePath = "";
    // Check input validity
    if (argc < 2) {
        std::cout << "Enter video file path: ";
        std::getline(std::cin, videoFilePath);
    }
    else {
        videoFilePath = argv[1];
    }

    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const auto rec = rerun::RecordingStream("rerun_example_cpp");
    // Try to spawn a new viewer instance.
    rec.spawn().exit_on_failure();

    // Create some data using the `grid` utility function.
    std::vector<rerun::Position3D> points = grid3d<rerun::Position3D, float>(-10.f, 10.f, 10);
    std::vector<rerun::Color> colors = grid3d<rerun::Color, uint8_t>(0, 255, 10);

    // Log the "my_points" entity with our data, using the `Points3D` archetype.
    rec.log("my_points", rerun::Points3D(points).with_colors(colors).with_radii({ 0.5f }));

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

    // Process frames
    SpatialConvolution spatial;
    TemporalConvolution temporalSmooth(kernelSizeSmoothing);
    TemporalConvolution temporalDerivative(kernelSizeDerivative);
    Mat frame, smoothedT, smoothedTX, smoothedTXY, I_t, I_x, I_y, derivativeDisplay;
    while (capture.read(frame)) {
        // Display original image
        imshow("Original image", frame);
        waitKey(1);

        smoothedT = temporalSmooth.convolve(frame, *gaussianKernel);
        if (!smoothedT.empty()) {
            imshow("Smoothed T", smoothedT);
            waitKey(1);
        }

        smoothedTX = spatial.convolveX(smoothedT, *gaussianKernel);
        if (!smoothedTX.empty()) {
            imshow("Smoothed TX", smoothedTX);
            waitKey(1);
        }

        smoothedTXY = spatial.convolveY(smoothedTX, *gaussianKernel);
        if (!smoothedTXY.empty()) {
            imshow("Smoothed TXY", smoothedTXY);
            waitKey(1);
        }
        
        I_t = temporalDerivative.convolve(smoothedTXY, derivativeKernel);
        if (!I_t.empty()) {
            normalize(I_t, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_t", derivativeDisplay);
            waitKey(1);
        }

        I_x = spatial.convolveX(smoothedTXY, derivativeKernel);
        if (!I_x.empty()) {
            normalize(I_x, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_x", derivativeDisplay);
            waitKey(1);
        }

        I_y = spatial.convolveY(smoothedTXY, derivativeKernel);
        if (!I_y.empty()) {
            normalize(I_y, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_y", derivativeDisplay);
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