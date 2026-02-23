#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define SIZE 16
#define BLOCK_DIM 4

__global__ void transpose(double *idata, double *odata, int width, int height)
{
    __shared__ double block[BLOCK_DIM][BLOCK_DIM + 1];

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

void hostTranspose(const double* in, double* out, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            out[x * height + y] = in[y * width + x];
        }
    }
}

void initMatrix(double* mat, int width, int height)
{
    for (int i = 0; i < width * height; i++)
    {
        mat[i] = i;
    }
}

bool checkResult(const double* a, const double* b, int size)
{
    const double epsilon = 1e-8;

    for (int i = 0; i < size; i++)
    {
        if (abs(a[i] - b[i]) > epsilon)
        {
            cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << endl;
            return false;
        }
    }

    return true;
}

void printMatrix(const double* mat, int width, int height, string msg)
{
    if (width <= 16 && height <= 16)
    {
        cout << "---------------  " << msg << "  ---------------" << endl;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cout << mat[y * width + x] << "\t";
            }
            cout << endl;
        }
    }
}

void handleError(cudaError_t error)
{
    if (error == cudaSuccess)
        return;

    switch (error)
    {
        case cudaErrorMemoryAllocation:
            cerr << "CUDA Error: Memory allocation failed" << endl;
            break;
        case cudaErrorInvalidValue:
            cerr << "CUDA Error: Invalid value" << endl;
            break;
        case cudaErrorInvalidDevice:
            cerr << "CUDA Error: Invalid device" << endl;
            break;
        case cudaErrorInvalidMemcpyDirection:
            cerr << "CUDA Error: Invalid memcpy direction" << endl;
            break;
        case cudaErrorLaunchFailure:
            cerr << "CUDA Error: Kernel launch failure" << endl;
            break;
        case cudaErrorLaunchTimeout:
            cerr << "CUDA Error: Kernel launch timeout" << endl;
            break;
        case cudaErrorLaunchOutOfResources:
            cerr << "CUDA Error: Kernel launch out of resources" << endl;
            break;
        case cudaErrorInitializationError:
            cerr << "CUDA Error: Initialization error" << endl;
            break;
        case cudaErrorMemoryValueTooLarge:
            cerr << "CUDA Error: Memory value too large" << endl;
            break;

        default:
            cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;
            break;
    }

    exit(EXIT_FAILURE);
}

__host__ void deviceTranspose(double *h_idata, double *h_odata, int width, int height, size_t bytes, float* millisecondsMalloc, float* millisecondsMemcpy, float* millisecondsKernel)
{
    cudaEvent_t start, stop;
    handleError(cudaEventCreate(&start));
    handleError(cudaEventCreate(&stop));

    double *d_idata, *d_odata;

    handleError(cudaEventRecord(start));

    handleError(cudaMalloc(&d_idata, bytes));
    handleError(cudaMalloc(&d_odata, bytes));

    handleError(cudaEventRecord(stop));
    handleError(cudaEventSynchronize(stop));

    handleError(cudaEventElapsedTime(millisecondsMalloc, start, stop));

    handleError(cudaEventRecord(start));

    handleError(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    handleError(cudaEventRecord(stop));
    handleError(cudaEventSynchronize(stop));

    handleError(cudaEventElapsedTime(millisecondsMemcpy, start, stop));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM);

    handleError(cudaEventRecord(start));

    transpose<<<grid, block>>>(d_idata, d_odata, width, height);

    handleError(cudaGetLastError());

    handleError(cudaEventRecord(stop));
    handleError(cudaEventSynchronize(stop));

    handleError(cudaEventElapsedTime(millisecondsKernel, start, stop));

    handleError(cudaDeviceSynchronize());

    handleError(cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost));

    handleError(cudaFree(d_idata));
    handleError(cudaFree(d_odata));

    handleError(cudaEventDestroy(start));
    handleError(cudaEventDestroy(stop));
}

void checkDevice()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0)
    {
        cerr << "No CUDA devices found or error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    int device = 0;
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);

    if (err != cudaSuccess)
    {
        cerr << "Failed to get device properties: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Using CUDA Device #" << device << ": " << deviceProp.name << endl;
    cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "  Total global memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
    cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << endl;
    cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "  Max threads dim: [" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << "]" << endl;
    cout << "  Max grid size: [" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << "]" << endl;

    cudaSetDevice(device);
}

int main(int argc, char** argv)
{
    int N = SIZE;

    if (argc > 1)
    {
        N = atoi(argv[1]);

        if (N <= 0)
        {
            cout << "Nevalidna dimenzija matrice, koristi default 16" << endl;
            N = SIZE;
        }
    }

    cout << "Dimenzija matrice: " << N << " x " << N << endl;

    const int width = N;
    const int height = N;

    const int size = width * height;
    const size_t bytes = size * sizeof(double);

    double* h_idata = new double[size];
    double* h_odata = new double[size];
    double* h_odata_cpu = new double[size];

    checkDevice();

    initMatrix(h_idata, width, height);

    printMatrix(h_idata, width, height, "Initial");

    float millisecondsMalloc = 0;
    float millisecondsMemcpy = 0;
    float millisecondsKernel = 0;

    deviceTranspose(h_idata, h_odata, width, height, bytes, &millisecondsMalloc, &millisecondsMemcpy, &millisecondsKernel);

    hostTranspose(h_idata, h_odata_cpu, width, height);

    printMatrix(h_odata, width, height, "GPU");
    printMatrix(h_odata_cpu, width, height, "CPU");

    bool result = checkResult(h_odata_cpu, h_odata, size);

    cout << (result ? "Transpose successful, results match!" : "Transpose failed, results do not match.") << endl;

    double numOps = 2.0 * width * height;

    double gflops = (numOps / (millisecondsKernel / 1000.0)) / 1e9;

    cout << "Allocation time: " << millisecondsMalloc << " ms" << endl;
    cout << "Memory copy time: " << millisecondsMemcpy << " ms" << endl;
    cout << "Kernel execution time: " << millisecondsKernel << " ms" << endl;
    cout << "Performance: " << gflops << " GFLOPS" << " (Larger matrices yield better performance)" << endl;

    delete[] h_idata;
    delete[] h_odata;
    delete[] h_odata_cpu;

    return 0;
}
