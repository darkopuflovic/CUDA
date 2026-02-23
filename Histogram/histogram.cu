#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

#define NUM_ELEMENTS 1 << 20
#define BIN_COUNT 256
#define THREADS_PER_BLOCK 256

// CPU histogram
void histogramCPU(const int *data, int *hist, int n)
{
    for (int i = 0; i < n; i++)
    {
        hist[data[i]]++;
    }
}

// CUDA histogram - globalna memorija
__global__ void histogramSimpleKernel(const int *data, int *hist, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Možda i while, ali nema potrebe za tim, pošto nema dovoljno memorije da bi jedna nit
    // izvršavala više od jednog elementa
    if (i < n)
    {
        // Koristimo atomicAdd zato što niti upisuju u isti niz
        // atomicAdd nad globalnom memorijom je jako spor
        atomicAdd(&hist[data[i]], 1);
    }
}

// CUDA histogram - shared memorija
__global__ void histogramSharedKernel(const int *data, int *hist, int n)
{
    __shared__ unsigned int localHist[BIN_COUNT];

    int tid = threadIdx.x;

    if (tid < BIN_COUNT)
    {
        localHist[tid] = 0;
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;

    // Možda može i while petlja umesto if, ali nema potrebe, kao i u prvom kernelu
    if (i < n)
    {
        // Ovde nad shared memorijom je mnogo brži
        atomicAdd(&localHist[data[i]], 1);
    }

    __syncthreads();

    if (tid < BIN_COUNT)
    {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}

// Poređenje
bool compareHistograms(const int *h1, const int *h2, int bins)
{
    for (int i = 0; i < bins; i++)
    {
        if (h1[i] != h2[i])
        {
            printf("Mismatch at bin %d: CPU=%d, GPU=%d\n", i, h1[i], h2[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    __int128_t N = (__int128_t)NUM_ELEMENTS;

    if (argc > 1)
    {
        N = (__int128_t)atoi(argv[1]);

        if (N <= 0)
        {
            printf("Nevalidna dimenzija matrice, koristi default.\n");
            N = (__int128_t)NUM_ELEMENTS;
        }
    }

    // Alokacija memorije za host
    int *h_data = (int*)malloc(N * sizeof(int));
    int *h_hist_cpu = (int*)calloc(BIN_COUNT, sizeof(int));
    int *h_hist_simple = (int*)calloc(BIN_COUNT, sizeof(int));
    int *h_hist_shared = (int*)calloc(BIN_COUNT, sizeof(int));

    // Random
    srand(time(NULL));

    for (int i = 0; i < N; i++)
    {
        h_data[i] = rand() % BIN_COUNT;
    }

    // CPU histogram
    printf("CPU histogram...\n");
    auto cpu_start = high_resolution_clock::now();
    histogramCPU(h_data, h_hist_cpu, N);
    auto cpu_end = high_resolution_clock::now();
    double cpu_time = duration<double, milli>(cpu_end - cpu_start).count();
    printf("CPU time: %.3f ms\n", cpu_time);

    // Alociranje GPU memorije
    int *d_data, *d_hist;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_hist, BIN_COUNT * sizeof(int));

    // Kopiranje niza u GPU memoriju
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Block i grid dimenzije
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + block.x - 1) / block.x);

    // Histogram GPU
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));
    cudaEventRecord(start);
    histogramSimpleKernel<<<grid, block>>>(d_data, d_hist, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeSimple = 0;
    cudaEventElapsedTime(&timeSimple, start, stop);
    cudaMemcpy(h_hist_simple, d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // Histogram GPU - shared memorija
    cudaMemset(d_hist, 0, BIN_COUNT * sizeof(int));
    cudaEventRecord(start);
    histogramSharedKernel<<<grid, block>>>(d_data, d_hist, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeShared = 0;
    cudaEventElapsedTime(&timeShared, start, stop);
    cudaMemcpy(h_hist_shared, d_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // Poređenje rezultata
    bool ok_simple = compareHistograms(h_hist_cpu, h_hist_simple, BIN_COUNT);
    bool ok_shared = compareHistograms(h_hist_cpu, h_hist_shared, BIN_COUNT);

    printf("\nResult check:\n");
    printf("Simple kernel: %s\n", ok_simple ? "OK" : "ERROR");
    printf("Shared kernel: %s\n", ok_shared ? "OK" : "ERROR");

    printf("\nTiming:\n");
    printf("Simple kernel: %.3f ms\n", timeSimple);
    printf("Shared kernel: %.3f ms\n", timeShared);

    // Brisanje
    free(h_data);
    free(h_hist_cpu);
    free(h_hist_simple);
    free(h_hist_shared);

    cudaFree(d_data);
    cudaFree(d_hist);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
