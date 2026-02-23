#include <iostream>

using namespace std;

#define SIZE 16
#define BLOCKSIZE 4

__device__ int compare(int a, int b)
{
    if (a == b)
    {
        return 1;
    }
    
    return 0;
}

__global__ void compute(int* d_in, int* d_out)
{
    d_out[threadIdx.x] = 0;

    for (int i = 0; i < SIZE / BLOCKSIZE; i++)
    {
        int val = d_in[i * BLOCKSIZE + threadIdx.x];
        d_out[threadIdx.x] += compare(val, 6);
    }
}

__host__ void call_gpu_compute(int *in_arr, int *out_arr)
{
    int *d_in_array, *d_out_array;

    cudaMalloc((void**)&d_in_array, SIZE * sizeof(int));
    cudaMalloc((void**)&d_out_array, BLOCKSIZE * sizeof(int));

    cudaMemcpy(d_in_array, in_arr, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    compute<<<1, BLOCKSIZE>>>(d_in_array, d_out_array);
    
    // Pošto je sledeća operacija cudaMemcpy, ona je sinhrona, pa nema potrebe za sinhronizacijom
    // Ali smo se mi obezbedili da se ne nastavi dok se asinhrona kernel funkcija ne završi
    // Nije najbolje za performanse
    cudaDeviceSynchronize();

    cudaMemcpy(out_arr, d_out_array, BLOCKSIZE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in_array);
    cudaFree(d_out_array);
}

int main()
{
    int *in_array, *out_array;
    int sum = 0;

    in_array = new int[SIZE] { 3, 6, 7, 5, 3, 5, 6, 2, 9, 1, 2, 7, 0, 9, 3, 6 };
    out_array = new int[BLOCKSIZE];

    call_gpu_compute(in_array, out_array);

    for (int i = 0; i < BLOCKSIZE; i++)
    {
        sum += out_array[i];
    }

    cout << "Cifra 6 se ponavlja " << sum << " puta." << endl;

    delete[] in_array;
    delete[] out_array;
}
