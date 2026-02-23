#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

#define SIZE 512
#define TILE_SIZE 32

// CPU množenje matrica
void matrixMulCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;

            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// Najprostiji GPU kernel: 1 nit računa 1 element, globalna memorija
__global__ void matMulKernelSimple(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// GPU kernel bez shared memorije (jednostavan tile pristup)
__global__ void matMulTileKernel(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.f;

    for (int tileIdx = 0; tileIdx < (N + TILE_SIZE - 1) / TILE_SIZE; tileIdx++)
    {
        // Računanje iz globalne memorije
        for (int k = 0; k < TILE_SIZE; k++)
        {
            int aCol = tileIdx * TILE_SIZE + k;
            int bRow = tileIdx * TILE_SIZE + k;

            if (row < N && aCol < N && bRow < N && col < N)
            {
                float a = A[row * N + aCol];
                float b = B[bRow * N + col];
                sum += a * b;
            }
        }
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

// GPU kernel sa shared memorijom (tile pristup)
__global__ void matMulTileKernelShared(const float *A, const float *B, float *C, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.f;

    for (int tileIdx = 0; tileIdx < (N + TILE_SIZE - 1) / TILE_SIZE; tileIdx++)
    {
        // Upis u shared memoriju
        if (row < N && tileIdx * TILE_SIZE + threadIdx.x < N)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * TILE_SIZE + threadIdx.x];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tileIdx * TILE_SIZE + threadIdx.y < N)
        {
            tileB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Čekanje da se sve upiše
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Upis nazad u globalnu memoriju
    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Funkcija za poređenje matrica
bool compareMatrices(const float *A, const float *B, int N)
{
    const float EPS = 1e-3f;

    for (int i = 0; i < N * N; i++)
    {
        if (fabs(A[i] - B[i]) > EPS)
        {
            cout << "Razlika na indeksu " << i
                 << ": CPU=" << fixed << setprecision(5) << A[i]
                 << ", GPU=" << B[i] << endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    int N = SIZE;

    if (argc > 1)
    {
        N = atoi(argv[1]);

        if (N <= 0)
        {
            cout << "Nevalidna dimenzija matrice, koristi default 16" << endl;
            N = 16;
        }
    }

    cout << "Dimenzija matrice: " << N << " x " << N << endl;

    size_t bytes = N * N * sizeof(float);

    // Alokacija CPU memorije
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_simple = (float*)malloc(bytes);
    float *h_C_tile = (float*)malloc(bytes);
    float *h_C_shared = (float*)malloc(bytes);

    // Inicijalizacija matrica
    srand(time(NULL));

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = (float)(rand() % 10 + 1);
        h_B[i] = (float)(rand() % 10 + 1);
    }

    // CPU množenje
    if (N <= SIZE)
    {
        cout << "CPU množenje matrica..." << endl;
        auto cpu_start = high_resolution_clock::now();
        matrixMulCPU(h_A, h_B, h_C_cpu, N);
        auto cpu_end = high_resolution_clock::now();
        double cpu_time = duration<double, milli>(cpu_end - cpu_start).count();
        cout << "CPU vreme: " << fixed << setprecision(3) << cpu_time << " ms" << endl;
    }

    // Alokacija na GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Kopiranje A i B na GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Block i grid dimenzije
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA event za merenje vremena
    cudaEvent_t start, stop;

    // 1. Najprostiji GPU kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulKernelSimple<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeSimple = 0;
    cudaEventElapsedTime(&timeSimple, start, stop);

    cudaMemcpy(h_C_simple, d_C, bytes, cudaMemcpyDeviceToHost);

    // 2. GPU kernel sa shared memorijom
    cudaEventRecord(start);
    matMulTileKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeTile = 0;
    cudaEventElapsedTime(&timeTile, start, stop);

    cudaMemcpy(h_C_tile, d_C, bytes, cudaMemcpyDeviceToHost);

    // 2. GPU kernel sa shared memorijom
    cudaEventRecord(start);
    matMulTileKernelShared<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeShared = 0;
    cudaEventElapsedTime(&timeShared, start, stop);

    cudaMemcpy(h_C_shared, d_C, bytes, cudaMemcpyDeviceToHost);

    bool ok_simple = true;
    bool ok_tile = true;
    bool ok_shared = true;

    // Provera rezultata
    if (N <= SIZE)
    {
        ok_simple = compareMatrices(h_C_cpu, h_C_simple, N);
        ok_tile = compareMatrices(h_C_cpu, h_C_tile, N);
        ok_shared = compareMatrices(h_C_cpu, h_C_shared, N);
    }
    else
    {
        ok_simple = compareMatrices(h_C_simple, h_C_tile, N) && compareMatrices(h_C_simple, h_C_shared, N);
        ok_tile = ok_simple;
        ok_shared = ok_shared;
    }

    cout << endl << "Provera rezultata:" << endl;
    cout << "Simple kernel: " << (ok_simple ? "OK" : "GRESKA") << endl;
    cout << "Tile kernel: " << (ok_tile ? "OK" : "GRESKA") << endl;
    cout << "Shared kernel: " << (ok_shared ? "OK" : "GRESKA") << endl;

    cout << endl << "Vremena GPU kernel-a:" << endl;
    cout << fixed << setprecision(3);
    cout << "Simple kernel: " << timeSimple << " ms" << endl;
    cout << "Tile kernel: " << timeTile << " ms" << endl;
    cout << "Shared kernel: " << timeShared << " ms" << endl;

    // Oslobađanje memoriju
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_simple);
    free(h_C_shared);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
