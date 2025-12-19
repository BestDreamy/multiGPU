__global__ void kernel_write(float* p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        p[idx] = idx;
    }
}
