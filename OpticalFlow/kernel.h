#ifndef KERNEL_H
#define KERNEL_H

class Kernel {
public:
    Kernel(const float* values, const int size);
    ~Kernel();

    static Kernel* generateGaussian(int size, float sigma);

    int getSize() const;
    float get(int index) const;

    void print() const;

private:
    Kernel(int size);

    void generateGaussian(float sigma);

    int size;
    float* kernel;
};
#endif // KERNEL_H