#include <iostream>

using namespace std;

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
    cout << "  CUDA cores: " << deviceProp.multiProcessorCount * 128 << endl;
    cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "  Max threads dim: [" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << "]" << endl;
    cout << "  Max grid size: [" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", "
              << deviceProp.maxGridSize[2] << "]" << endl;

    err = cudaSetDevice(device);

    if (err != cudaSuccess)
    {
        cerr << "Failed to set device: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    checkDevice();

    return 0;
}
