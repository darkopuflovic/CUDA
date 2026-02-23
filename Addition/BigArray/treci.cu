#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define N 1000000

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    int *a = new int[N], *b = new int[N], *c = new int[N];

    int *d_a, *d_b, *d_c;

    srand(time(0));

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 10 + 1;
        b[i] = rand() % 10 + 1;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid,threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    bool valid = true;

    for (int i = 0; i < N; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            valid = false;
        }
    }

    cout << "The results of all " << N << " additions are " << (valid ? "valid." : "invalid.") << endl;

    delete[] c;
    delete[] b;
    delete[] a;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
