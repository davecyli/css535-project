/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Parent class for TemporalConvolution and SpatialConvolution. Provides constructor behavior and
defines the implementation to be used by derived classes. Also provides convert() method for
common preprocessing.
------------------------------------------------------------------------------------------------ */

#include "convolution.h"
#include "cuda_runtime.h"

/**
 * @brief Converts input frames to grayscale and float format
 *
 * @param frame: The input frame to be converted
 * 
 * @return The converted frame
 */
Mat Convolution::convert(const Mat& frame) const {
    // Check for empty frames
    if (frame.empty()) return frame;

    // Copy and convert to grayscale and float if needed
    Mat converted;
    if (frame.channels() == 3) {
        // Convert color images to grayscale
        cvtColor(frame, converted, COLOR_BGR2GRAY);
        // Normalize to 0.0-1.0 range as floating point
        converted.convertTo(converted, CV_32F, 1.0 / 255.0);
    }
    else {
        // If already converted, return original frame
        return frame;
    }
    return converted;
}