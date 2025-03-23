#include "utilities.h"

/**
 * Converts optical flow vectors (flowX, flowY) to a color-coded visualization using HSV color mapping.
 * The resulting image uses hue to represent direction and value (brightness) to represent magnitude.
 */
cv::Mat flowToHSV(const cv::Mat& flowX, const cv::Mat& flowY, cv::Mat& bgr) {
    // Convert the algorithm's output into Polar coordinates
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flowX, flowY, magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    // Build hsv image
    cv::Mat _hsv[3], hsv, hsv8;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    cv::normalize(hsv, hsv, 0.0f, 1.0f, cv::NORM_MINMAX);
    hsv.convertTo(hsv8, CV_8U, 255.0);

    // Convert to BGR for display
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    
    return bgr;
}