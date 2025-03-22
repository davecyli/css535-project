#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>

/**
 * Converts optical flow vectors (flowX, flowY) to a color-coded visualization using HSV color mapping.
 * 
 * @param flowX Matrix containing the x-component of the optical flow vectors
 * @param flowY Matrix containing the y-component of the optical flow vectors
 * @return Color-coded BGR image representing the flow vectors
 */
cv::Mat convertFlowToColorMap(const cv::Mat& flowX, const cv::Mat& flowY, cv::Mat&bgr);

#endif // UTILITIES_H