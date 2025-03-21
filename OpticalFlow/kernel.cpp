/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Generates and manages access to a 1D kernel for convolution with specialized support for
generating Gaussian kernels
------------------------------------------------------------------------------------------------ */

#include "kernel.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef PI
#define PI 3.141593f
#endif

/**
 * @brief Constructor that creates a kernel from provided values
 *
 * @param values: Array of kernel coefficient values
 * @param size: Number of elements in the kernel
 */
Kernel::Kernel(const float* values, const int size) : size(size) {
    // Allocate memory for kernel coefficients
    kernel = new float[size];
    // Copy provided values into the kernel
    memcpy(kernel, values, size * sizeof(float));
}

/**
 * @brief Constructor that only allocates memory. Used by Gaussian factory method.
 *
 * @param size: Number of elements in the kernel
 */
Kernel::Kernel(int size) : size(size) {
    // Allocate and zero-initialize memory for kernel
    kernel = new float[size]();
}

/**
 * @brief Destructor for cleaning up dynamically allocated memory
 */
Kernel::~Kernel() {
    // Free dynamically allocated memory
    delete[] kernel;
}

/**
 * @brief Factory method to create a Gaussian kernel
 *
 * @param size: Size of the kernel (should be odd)
 * @param sigma: Standard deviation of the Gaussian distribution
 * @return Pointer to a newly allocated Gaussian kernel
 */
Kernel* Kernel::generateGaussian(int size, float sigma) {
    // Create an empty kernel
    Kernel* k = new Kernel(size);
    // Fill it with Gaussian values
    k->generateGaussian(sigma);
    return k;
}

/**
 * @brief Fills the kernel with Gaussian values
 *
 * @param sigma Standard deviation controlling the spread of the Gaussian
 */
void Kernel::generateGaussian(float sigma) {
    int half = size / 2;
    float sum = 0.0;

    // Generate 1D Gaussian kernel
    for (int i = 0; i <= half; i++) {
        // Calculate Gaussian function value
        float value = exp(-(i * i) / (2.0f * sigma * sigma)) / (sqrt(2.0f * PI) * sigma);

        // Set symmetric values on both sides of center
        kernel[half + i] = value;
        kernel[half - i] = value;

        // Accumulate sum for normalization (avoiding double-counting center)
        sum += (i == 0) ? value : 2 * value;
    }
    // Normalize kernel to ensure sum equals 1
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

/**
 * @brief Gets the size of the kernel
 *
 * @return Number of elements in the kernel
 */
int Kernel::getSize() const {
    return size;
}

/**
 * @brief Accesses a specific kernel element
 *
 * @param index: Index of the element to access
 * 
 * @return Value of the kernel at the specified index
 */
float Kernel::getElement(int index) const {
    return kernel[index];
}

/**
 * @brief Gets direct access to raw kernel data
 *
 * @return Pointer to the kernel's internal data array
 */
float* Kernel::getRawData() const {
    return kernel;
}

/**
 * @brief Prints the kernel values to standard output
 */
void Kernel::print() const {
    // Print each kernel coefficient with high precision
    for (int i = 0; i < size; i++) {
        printf("%.6f ", kernel[i]);
    }
    printf("\n");
}