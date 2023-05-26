#include <stdio.h>
#include <cassert> // ???
#include <iostream>
#include <cmath>

void init_matrices(int* a, int* b, int n)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            a[i*n + j] = rand() % 100;
            b[i*n + j] = rand() % 100;
        }
    }
}


// Launch CUDA kernel
__global__ void matrixMul(int* a, int* b, int* c, int n)
{
    // Compute each thread's row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute each thread's column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    // Boundary protection
    if((row < n) && (col < n))
    {
        // Iterate over row and column
        for(int k = 0; k<n; k++)
        {
            // Accumulate the result for single element
            temp_sum += a[row * n + k] * b[k * n + col];
        }
        // Assign result
        c[row * n + col] = temp_sum;
    }
}

// Check result on the CPU
void verify_result(int* a, int* b, int* c, int n) {
  // For every row...
  for (int i = 0; i < n; i++) {
    // For every column...
    for (int j = 0; j < n; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < n; k++) {
        // Accumulate the partial results
        tmp += a[i * n + k] * b[k * n + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * n + j]);
    }
  }
}


int main()
{
    // Matrix size of 1024 x 1024
    int n = pow(2, 10);

    // Size(in bytes) of matrix
    size_t bytes = n*n*sizeof(int);

    // Host pointers
    int *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocated device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initializes matrices
    init_matrices(h_a, h_b, n);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threads per block
    int BLOCK_SIZE = 16;

    // Block in each dimension
    int GRID_SIZE = (int)ceil(n/BLOCK_SIZE);

    // Use dim3 objects
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    matrixMul<<<grid, threads>>>(d_a, d_b, d_c, n);

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c, n);

    return 0;
}

