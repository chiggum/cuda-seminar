#include <stdio.h>
#include "gputimer.h"

const int ARRAY_SIZE = 10000;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(long double);

void sequential_square_kernel(long double *d_in, long double *d_out)
{
    int i;
    for(i = 0; i < ARRAY_SIZE; ++i)
    {
        d_out[i] = d_in[i] * d_in[i];
    }
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
        
    GpuTimer timer;
    timer.Start();
    
    // launch the kernel
    sequential_square_kernel(h_in, h_out);
   
    timer.Stop();
    printf("%f msecs.\n", timer.Elapsed());
      
	return 0;
}