#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int i = threadIdx.x;

    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int a[N];
    int b[N];
    int c[N];

    int *d_a, *d_b, *d_c;

    srand(time(0));

    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 10 + 1;
        b[i] = rand() % 10 + 1;
    }

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, &a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, N * sizeof(int), cudaMemcpyHostToDevice);

    void* args[] = {&d_a, &d_b, &d_c};

    cudaLaunchKernel((void*)add, dim3(1), dim3(N), args, 0, 0);

    cudaMemcpy(&c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Result of addition is: " << endl;

    for (int i = 0; i < N; i++)
    {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
