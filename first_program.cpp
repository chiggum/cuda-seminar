#include <stdio.h>

const int ARRAY_SIZE = 100;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

__global__ void square(float *d_in, float *d_out)
{
    int tid = threadIdx.x;
    d_out[tid] = d_in[tid] * d_in[tid];.
}

int main() 
{
    // generating array on HOST
    float h_in[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_in[i] = float(i);
    }
    
    // declaring GPU Memory pointers
    float *d_in;
    float *d_out;
    
    // allocating memory on GPU
    cudaMalloc(&d_in, ARRAY_BYTES);
    cudaMalloc(&d_out, ARRAY_BYTES);
    
    // transfer the HOST input array to GPU (DEVICE)
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch the kernel
    square<<<1, ARRAY_SIZE>>>(d_in, d_out);
    
    // Copying the result Back to CPU (HOST)
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    
    // Print out the resulting array
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        printf("%f\n", h_out[i]);
    }
    
    // freeing the pointers
    cudaFree(d_in);
    cudaFree(d_out);

   
    return 0;
}