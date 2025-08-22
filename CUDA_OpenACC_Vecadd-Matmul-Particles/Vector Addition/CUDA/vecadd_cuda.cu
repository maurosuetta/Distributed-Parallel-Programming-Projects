#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// BLOCKSIZE
#define BLOCKSIZE 128

// CUDA ERROR CHECK
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void vecadd_cuda(double *A, double *B, double *C, const int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    int N;

    if (argc != 2)
    {
        printf("Usage: %s <vector size N>\n", argv[0]);
        return 1;
    }
    else
    {
        N = atoi(argv[1]);
    }
    printf("Vector size: %d\n", N);

    //
    // Memory allocation
    //
    size_t array_size = N * sizeof(double);

    // Host
#ifndef PINNED
    double *h_A = (double *)malloc(array_size);
    double *h_B = (double *)malloc(array_size);
    double *h_C = (double *)malloc(array_size);
#else
    double *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void **)&h_A, array_size));
    CUDA_CHECK(cudaMallocHost((void **)&h_B, array_size));
    CUDA_CHECK(cudaMallocHost((void **)&h_C, array_size));
#endif

    // Device
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, array_size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, array_size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, array_size));
    CUDA_CHECK(cudaMemset(d_C, 0, array_size)); // Init d_C to 0

    //
    // Create random values
    //
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (double)i;
        h_B[i] = (double)(2 * (N - i));
    }

    //
    // Vector addition kernel
    //

    cudaEvent_t start, end;
    float total_time_ms = 0.0;
    float time_ms = 0.0;

    // Create CUDA events
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));;

    // Copy arrays host to device
    CUDA_CHECK(cudaEventRecord(start)); // start timing copy to Device
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, array_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(end)); // End timing copy to Device
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));

    printf("Copy A and B Host to Device elapsed time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;
    time_ms = 0.0;

    CUDA_CHECK(cudaEventRecord(start)); // Start timing kernel

    // Calculate number of thread blocks
    dim3 threadsPerBlock(BLOCKSIZE, 1, 1);
    dim3 blocks((N + BLOCKSIZE - 1) / BLOCKSIZE, 1, 1);

    vecadd_cuda<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); // Check kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check if there was an error during the kernel

    CUDA_CHECK(cudaEventRecord(end)); // End timing kernel
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));

    printf("Kernel elapsed time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;
    time_ms = 0.0;


    // Copy back results to host
    CUDA_CHECK(cudaEventRecord(start)); // Start timing copy to Host

    CUDA_CHECK(cudaMemcpy(h_C, d_C, array_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(end)); // End timing copy to Host
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));

    printf("Copy C Device to Host elapsed time: %.9f seconds\n", time_ms / 1000);
    total_time_ms += time_ms;

    // Total time
    printf("Total elapsed time: %.9f seconds\n", total_time_ms / 1000);


    //
    // Validation
    //
    for (int i = 0; i < N; i++)
    {
        double expected = (double)2 * N - i;
        double local_err = fabs(expected - h_C[i]);
        if (local_err > 1.0e-6)
        {
            printf("Error at i = %d: fabs( %f - c[%d] ) = %e > %e\n", i, expected, i, local_err, 1.0e-6);
            return 1;
        }
    }

    //
    // Free memory
    //
    // Host
#ifndef PINNED
    free(h_A);
    free(h_B);
    free(h_C);
#else
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
#endif

    // Device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}