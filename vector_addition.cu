#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cmath>

// CUDA kernel launch
__global__ void vectorAdd(int* a, int* b, int* c, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

// Check the correction of the result
void checkResult(int* a, int* b, int* c, int n)
{
    for(int i = 0; i<n; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }
}

// Initialize random matrix with values 0-99
void initMatrix(int* a, int n)
{
    for(int i = 0; i<n; i++)
    {
        a[i] = rand() % 100;
    }
}

int main()
{
    // Vector size of 2^16
    int n = pow(2, 16);

    // Host vector pointers
    int *h_a, *h_b, *h_c;
    // Device vector pointers
    int *d_a, *d_b, *d_c;
    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    // Allocate host memory 
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize vectors a and b
    initMatrix(h_a, n);
    initMatrix(h_b, n);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size
    int NUM_THREADS = 256;

    // Grid Size
    int NUM_BLOCKS = (int)ceil(n/NUM_THREADS);

    // Launch kernel on default stream w/o shmem
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Check result for errors
    checkResult(h_a, h_b, h_c, n);

    return 0;

}