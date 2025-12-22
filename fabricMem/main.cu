#include <stdio.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int uvaSupported = 0;
    cudaDeviceGetAttribute(&uvaSupported, cudaDevAttrUnifiedAddressing, 0);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 8) {
        printf("Need at least 8 GPUs\n");
        return 0;
    }

	int cudaDev;
    cudaSetDevice(7);
    cudaError_t err = cudaGetDevice(&cudaDev);
    printf("GPU%d in runtime\n", cudaDev);

    CUdevice realDev;
    CUresult result = cuDeviceGet(&realDev, cudaDev);
    printf("GPU%d in driver\n", realDev);

    int flag = 0;
    result = cuDeviceGetAttribute(
        &flag, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        realDev);
    assert(flag == 1);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // Check if GPU supports fabric handle
    result = cuDeviceGetAttribute(
        &flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        realDev);
    assert(flag == 1);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    /*
    1. CU_MEM_LOCATION_TYPE_DEVICE: HBM (Fabric memory only)
    2. CU_MEM_LOCATION_TYPE_HOST: DDR
    3. CU_MEM_LOCATION_TYPE_HOST_NUMA
    */
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = realDev;
    // Check if GPU supports GPU-direct RDMA with CUDA VMM
    result = cuDeviceGetAttribute(
        &flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
        realDev);
    if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

    uint64_t granularity = 0;
    result = cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    printf("Minimum allocation granularity: %ldMB\n", granularity / 1024 / 1024);
    
    uint64_t fp_size = 2ll * 1024 * 1024; // 2MB
    fp_size = (fp_size + granularity - 1) & ~(granularity - 1);
    if (fp_size == 0) fp_size = granularity;

    uint64_t free_size = 0, total_size = 0;
    cuMemGetInfo(&free_size, &total_size);
    printf("Free memory: %ldMB, Total memory: %ldMB\n", free_size / 1024 / 1024, total_size / 1024 / 1024);

    CUmemGenericAllocationHandle handle;
    result = cuMemCreate(&handle, fp_size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        printf("cuMemCreate failed: %d\n", result);
        exit(0);
    }

    cuMemGetInfo(&free_size, &total_size);
    printf("Free memory: %ldMB, Total memory: %ldMB\n", free_size / 1024 / 1024, total_size / 1024 / 1024);

    // CUdeviceptr base;
    // size_t size;
    // cuMemGetAddressRange(&base, &size, (CUdeviceptr)ptr);

    void *ptr = nullptr;
    // Allocate virtual address space
    result = cuMemAddressReserve((CUdeviceptr *)&ptr, fp_size, granularity, 0, 0);
    if (result != CUDA_SUCCESS) {
        printf("cuMemAddressReserve failed: %d\n", result);
        cuMemRelease(handle);
        exit(0);
    }
    result = cuMemMap((CUdeviceptr)ptr, fp_size, 0, handle, 0);
    if (result != CUDA_SUCCESS) {
        printf("cuMemMap failed: %d\n", result);
        cuMemAddressFree((CUdeviceptr)ptr, fp_size);
        cuMemRelease(handle);
        exit(0);
    }
    
}
