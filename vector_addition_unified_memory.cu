#include <stdio.h>
#include <cassert> // ???
#include <iostream>
#include <cmath>

// CUDA kernel for vector addition
__global__ void vectorAddUM(int *a, int *b, int *c, int N)
{
    // blockDim: size of the thread block
	// blockIdx : id of the block
	// threadIdx: offset of the thread inside the block
	// Calculate global thread thread ID
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	// Boundary check
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main()
{
    // Get the device ID for other CUDA calls
    int id = cudaGetDevice(&id);

	// Array size of 2^16 
	const int n = pow(2, 16);
	size_t bytes = n * sizeof(int);

	// Declare unified memory pointers
	int *a, *b, *c;

	// Allocation memory for these pointers
    // Unified memory is a powerful thing 
    // when we don't have to explicitly manage memory 
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);
	
	// Initialize vectors
	for(int i=0; i<n; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Threads per CTA
    int BLOCK_SIZE = pow(2, 10);

    // CTAs per Grid
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    // Call CUDA kernel
    // GPU starts up and it says i dont have any of this data
    // So it does something called a page fault(page in memory from CPU to GPU)
    // We can hints to say when data should be where
    // We do this by prefetching
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);
    vectorAddUM<<<GRID_SIZE, BLOCK_SIZE>>>(a,b,c,n);

    // Wait for all previous operations before using values
    cudaDeviceSynchronize();

    // We can also prefetch the result back to CPU
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // Verify the result on the CPU
    for(int i=0; i<n; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }

    // Free unified memory(same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}


