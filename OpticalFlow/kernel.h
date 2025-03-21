/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Generates and manages access to a 1D kernel for convolution with specialized support for 
generating Gaussian kernels
------------------------------------------------------------------------------------------------ */

#ifndef KERNEL_H
#define KERNEL_H

class Kernel {
public:
    Kernel(const float* values, const int size);
    ~Kernel();

    static Kernel* generateGaussian(int size, float sigma);

    int getSize() const;
    float getElement(int index) const;
    float* getRawData() const;

    void print() const;

private:
    Kernel(int size);

    void generateGaussian(float sigma);

    int size;
    float* kernel;
};
#endif // KERNEL_H