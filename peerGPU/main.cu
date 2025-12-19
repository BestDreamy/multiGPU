#include <cuda_runtime.h>
#include <stdio.h>
#include "kernel.cu"

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs\n");
        return 0;
    }

    const int N = 256;
    float* d_ptr = nullptr;

    cudaSetDevice(1);
    cudaMalloc(&d_ptr, N * sizeof(float));

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    kernel_write<<<1, 256>>>(d_ptr, N);
    cudaDeviceSynchronize();

    cudaSetDevice(1);

    float h[N];
    cudaMemcpy(h, d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i) {
        printf("h[%d] = %f\n", i, h[i]);
    }

    cudaFree(d_ptr);
    return 0;
}
