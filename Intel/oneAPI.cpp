#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

constexpr size_t N = 1024;

int main() {
    // Create SYCL queue to target the default device (GPU/CPU/FPGA)
    queue q;

    // Allocate and initialize host memory
    int *A = malloc_shared<int>(N, q);
    int *B = malloc_shared<int>(N, q);
    int *C = malloc_shared<int>(N, q);

    for (size_t i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Submit kernel to device
    q.parallel_for(range<1>(N), [=](id<1> i) {
        C[i] = A[i] + B[i];
    }).wait();

    // Print first few results
    for (int i = 0; i < 10; ++i) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Free memory
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}
