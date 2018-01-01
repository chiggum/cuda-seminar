#include <stdio.h>
#include "gputimer.h"

const int ARRAY_SIZE = 10000;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(long double);

__global__ void square_kernel(long double *d_in, long double *d_out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < ARRAY_SIZE)
        d_out[tid] = d_in[tid] * d_in[tid];
}

int main() 
{
    // generating array on HOST
    long double h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_in[i] = i;
    }
    long double h_out[ARRAY_SIZE];
    
    // declaring GPU Memory pointers
    long double *d_in;
    long double *d_out;
    
    // allocating memory on GPU
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    
    // transfer the HOST input array to GPU (DEVICE)
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    GpuTimer timer;
    timer.Start();
    
    // launch the kernel
    square_kernel<<<10, ARRAY_SIZE / 10>>>(d_in, d_out);
   
    timer.Stop();
    printf("%f msecs.\n", timer.Elapsed());
    
    // Copying the result Back to CPU (HOST)
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    
    // freeing the pointers
    cudaFree(d_in);
    cudaFree(d_out);

   
    return 0;
}