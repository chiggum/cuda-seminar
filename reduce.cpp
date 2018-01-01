/*
 * Non optimized reduce global access
 */
#include <stdio.h>

const int ARRAY_SIZE = 4096;  // only powers of 2
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

__global__ void reduceToSum(float *d_in, float *d_out)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    for(int s = blockDim.x / 2; s > 0; s = s / 2)
    {
        if(tid < s)
        {
                d_in[myId] += d_in[myId + s];    
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }       
}

void sum(float *d_in, float *d_sum)
{
    const int blockSize = 1024;
    const int gridSize = (ARRAY_SIZE - 1)/ blockSize + 1;

    float *d_out;
    cudaMalloc(&d_out, gridSize * sizeof(float));

    // Launch Kernel
    reduceToSum<<< gridSize, blockSize >>>(d_in, d_out);

    reduceToSum<<< 1, gridSize >>>(d_out, d_sum);

    cudaFree(d_out);
}

int main() 
{
    // generating array on HOST
    float h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_in[i] = float(i);
    }
    float h_out;
    
    // declaring GPU Memory pointers
    float *d_in;
    float *d_sum;

   
    // allocating memory on GPU
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_sum, 1 * sizeof(float));
    
    // transfer the HOST input array to GPU (DEVICE)
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  
    sum(d_in, d_sum);

    // Copying the result Back to CPU (HOST)
    cudaMemcpy(&h_out, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print out the resulting sum
    printf("%f\n", h_out);
    
    
    // freeing the pointers
    cudaFree(d_in);
    cudaFree(d_sum);

   
    return 0;
}