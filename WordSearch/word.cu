#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_WORD 16

// Konstante su read-only i imaju jako ograničen prostor.
// Ali imaju svoj keš, koji je jako brz.
// Vidljive su svim nitima u celoj aplikaciji! Sve dok traje izvršenje ili
// se ne obriše.
// Konstante su optimizovane za broadcast. Ako je potrebno više podataka, šalju se jedan po jedan.
// https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
__constant__ char d_word_const[MAX_WORD];


// Bez konstante i bez shared memorije. Ipak, nije mnogo sporije, zato što će većina podataka biti u kešu.
__global__ void searchGlobal(char* text, char* word, int t_len, int w_len, int* count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= t_len - w_len)
    {
        int match = 1;

        for (int j = 0; j < w_len; j++)
        {
            if (text[i + j] != word[j])
            {
                match = 0;
                break;
            }
        }

        if (match)
        {
            atomicAdd(count, 1);
        }
    }
}

// Registri su privatni, tako da samo jedna nit može da im pristupi, što nije optimalno i može
// da se vidi po vremenu potrebnom za izvršenje.
// Još veći problem korišćenja registara može da bude veliki broj podataka koji u njih treba da se upiše.
// Ukoliko je to slučaj, oni podaci koji ne mogu da se upišu se upisuju u lokalnu memoriju, koja je jako spora.
__global__ void searchRegisters(char* text, char* word, int t_len, int w_len, int* count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    char target[MAX_WORD];

    for (int p = 0; p < w_len && p < MAX_WORD; p++)
    {
        target[p] = text[i + p];
    }

    if (i <= t_len - w_len)
    {
        int match = 1;

        for (int j = 0; j < w_len; j++)
        {
            if (target[j] != d_word_const[j])
            {
                match = 0;
                break;
            }
        }

        if (match)
        {
            atomicAdd(count, 1);
        }
    }
}

// Shared memory pristup, koji je neznatno brži. Razlog za to je keš koji globalna memorija koristi.
// Kada se podaci čitaju redom, keš će često imati odgovarajući podatak, pa će biti dosta brži od pristupa
// globalnoj memoriji svaki put.
// U slučaju da se pristupa nasumično podacima, ovaj pristup bi bio mnogo brži.
__global__ void searchShared(char* text, char* word, int t_len, int w_len, int* count)
{
    __shared__ char table[BLOCK_SIZE + MAX_WORD];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < t_len)
    {
        table[tid] = text[i];
    }

    if (tid < w_len - 1 && (i + blockDim.x) < t_len)
    {
        table[tid + blockDim.x] = text[i + blockDim.x];
    }

    __syncthreads();

    if (i <= t_len - w_len)
    {
        int match = 1;

        for (int j = 0; j < w_len; j++)
        {
            if (table[tid + j] != d_word_const[j])
            {
                match = 0;
                break;
            }
        }

        if (match)
        {
            atomicAdd(count, 1);
        }
    }
}

int main(int argc, char** argv)
{
    const int h_text_length = 104857600; // 100 MB

    if (argc <= 1)
    {
        printf("Unesite reč.\n");
        return -1;
    }

    const char* h_word = argv[1];
    int h_word_length = (int)strlen(h_word);

    char* h_text = (char*)malloc(h_text_length);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < h_text_length; i++)
    {
        h_text[i] = 'a' + (rand() % 26);
    }

    char *d_text, *d_word;
    int *d_count, h_global_result, h_register_result, h_shared_result;
    cudaMalloc(&d_text, h_text_length);
    cudaMalloc(&d_word, h_word_length);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_text, h_text, h_text_length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, h_word, h_word_length, cudaMemcpyHostToDevice);
    // Samo host može da kreira konstantu, i to ovako.
    cudaMemcpyToSymbol(d_word_const, h_word, h_word_length);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t_global, t_register, t_shared;

    int threads = BLOCK_SIZE;
    int blocks = (h_text_length + threads - 1) / threads;

    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(start);
    searchGlobal<<<blocks, threads>>>(d_text, d_word, h_text_length, h_word_length, d_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_global, start, stop);
    cudaMemcpy(&h_global_result, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(start);
    searchRegisters<<<blocks, threads>>>(d_text, d_word, h_text_length, h_word_length, d_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_register, start, stop);
    cudaMemcpy(&h_register_result, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(d_count, 0, sizeof(int));
    cudaEventRecord(start);
    searchShared<<<blocks, threads>>>(d_text, d_word, h_text_length, h_word_length, d_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t_shared, start, stop);
    cudaMemcpy(&h_shared_result, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    clock_t c_start = clock();
    int cpu_count = 0;

    for (int i = 0; i <= h_text_length - h_word_length; i++)
    {
        if (strncmp(h_text + i, h_word, h_word_length) == 0) cpu_count++;
    }

    clock_t c_end = clock();
    float t_cpu = (float)(c_end - c_start) * 1000.0f / CLOCKS_PER_SEC;

    printf("╔═══════════════════════╦══════════════╦══════════════╗\n");
    printf("║ %-21s ║ %-12s ║ %-12s ║\n", "Device", "Result", "Time (ms)");
    printf("╠═══════════════════════╬══════════════╬══════════════╣\n");
    printf("║ %-21s ║ %-12d ║ %-12.4f ║\n", "CPU (RAM)", cpu_count, t_cpu);
    printf("║ %-21s ║ %-12d ║ %-12.4f ║\n", "GPU (Global memory)", h_global_result, t_global);
    printf("║ %-21s ║ %-12d ║ %-12.4f ║\n", "GPU (Registers)", h_register_result, t_register);
    printf("║ %-21s ║ %-12d ║ %-12.4f ║\n", "GPU (Shared memory)", h_shared_result, t_shared);
    printf("╚═══════════════════════╩══════════════╩══════════════╝\n");

    free(h_text);
    cudaFree(d_text);
    cudaFree(d_word);
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
