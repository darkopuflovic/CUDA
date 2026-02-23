#include <iostream>
#include <hip/hip_runtime.h>

#define N 1024  // Size of arrays

// GPU Kernel for vector addition
__global__ void vector_add(const int* A, const int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

int main() {
    // Host arrays
    int *h_A, *h_B, *h_C;

    // Allocate host memory
    h_A = new int[N];
    h_B = new int[N];
    h_C = new int[N];

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Device arrays
    int *d_A, *d_B, *d_C;

    // Allocate device memory
    hipMalloc(&d_A, N * sizeof(int));
    hipMalloc(&d_B, N * sizeof(int));
    hipMalloc(&d_C, N * sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_A, h_A, N * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vector_add, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);

    // Copy result back to host
    hipMemcpy(h_C, d_C, N * sizeof(int), hipMemcpyDeviceToHost);

    // Print some results
    for (int i = 0; i < 10; ++i) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
