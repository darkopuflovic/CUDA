#include <iostream>
#include <vector>

using namespace std;

#define SIZE 256

__global__ void reduction_sum(int *array, int *result, int N)
{
    __shared__ int partial_sum[SIZE];
    int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (tid + blockDim.x < N)
    {
        partial_sum[threadIdx.x] = array[tid] + array[tid + blockDim.x];
    }
    else if (tid < N)
    {
        partial_sum[threadIdx.x] = array[tid];
    }
    else
    {
        partial_sum[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        result[blockIdx.x] = partial_sum[0];
    }
}

int main(int argc, char **argv)
{
    int N = 65536; // 2¹⁶

    if (argc > 1)
    {
        N = atoi(argv[1]);

        if (N <= 0)
        {
            cout << "Nevalidna dimenzija niza, koristimo default 65536" << endl;
            N = 65536;
        }
    }

    size_t bytes = N * sizeof(int);
    vector<int> vector_data(N);
    vector<int> vector_result(N);

    for (auto i = 0; i < N; i++)
    {
        vector_data[i] = 1;
        //vector_data[i] = rand() % 10;
    }

    int *device_vector_data, *device_vector_result;
    cudaMalloc(&device_vector_data, bytes);
    cudaMalloc(&device_vector_result, bytes);
    cudaMemcpy(device_vector_data, vector_data.data(), bytes, cudaMemcpyHostToDevice);

    const int TB_SIZE = SIZE;
    int GRID_SIZE = (N + TB_SIZE * 2 - 1) / TB_SIZE / 2;

    reduction_sum<<<GRID_SIZE, TB_SIZE>>>(device_vector_data, device_vector_result, N);
    reduction_sum<<<1, TB_SIZE>>>(device_vector_result, device_vector_result, TB_SIZE);

    cudaMemcpy(vector_result.data(), device_vector_result, bytes, cudaMemcpyDeviceToHost);

    cout << "SUM: " << vector_result[0] << endl;

    return 0;
}
