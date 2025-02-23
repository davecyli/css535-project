#include "kernel.h"
#include <cstdio>
#include <cmath>
#include <cstring>

#ifndef PI
#define PI 3.141593f
#endif

Kernel::Kernel(const float* values, const int size) : size(size) {
    kernel = new float[size];
    memcpy(kernel, values, size * sizeof(float));
}

Kernel::Kernel(int size) : size(size) {
    kernel = new float[size]();
}

Kernel::~Kernel() {
    delete[] kernel;
}

Kernel* Kernel::generateGaussian(int size, float sigma) {
    Kernel* k = new Kernel(size);
    k->generateGaussian(sigma);
    return k;
}

void Kernel::generateGaussian(float sigma) {
    int half = size / 2;
    float sum = 0.0;
 
    // Generate 1D Gaussian kernel
    for (int i = 0; i <= half; i++) {
        float value = exp(-(i * i) / (2.0f * sigma * sigma)) / (sqrt(2.0f * PI) * sigma);
        kernel[half + i] = value;
        kernel[half - i] = value;
        sum += (i == 0) ? value : 2 * value;
    }
    // Normalize
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

int Kernel::getSize() const {
    return size;
}

float Kernel::get(int index) const {
    return kernel[index];
}

void Kernel::print() const {
    for (int i = 0; i < size; i++) {
        printf("%.6f ", kernel[i]);
    }
    printf("\n");
}