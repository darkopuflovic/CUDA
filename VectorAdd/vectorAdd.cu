#include <iostream>

using namespace std;

#define N 512

__global__ void vector_add(const float *a, const float *b, float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    float *h_a, *h_b, *h_c_gpu, *h_c_cpu;
    float *d_a, *d_b, *d_c;

    size_t bytes = N * sizeof(float);

    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c_gpu = (float *)malloc(bytes);
    h_c_cpu = (float *)malloc(bytes);

    for (int i = 0; i < N; i++)
    {
        h_a[i] = float(i);
        h_b[i] = float(i * 2);
    }

    for (int i = 0; i < N; i++)
    {
        h_c_cpu[i] = h_a[i] + h_b[i];
    }

    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    cout << "A + B = GPU_Result (CPU_Result)" << endl;
    cout << "--------------------------------------------------------" << endl;

    for (int i = 0; i < N; i++)
    {
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c_gpu[i] << " (" << h_c_cpu[i] << ")" << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    return 0;
}
