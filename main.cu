#include "repeat.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include<string>
#include<string.h>
template<typename REAL>
__global__ void kernel(REAL p, REAL q, REAL *out, unsigned int *time_stamp){
    unsigned int id = threadIdx.x ;
    unsigned int start=0;
    unsigned int end=0;
    // unsigned int warp_id = id / 32;
    start=clock();
    repeat1024(p=p*q;q=p*q;); 
    end=clock();

    // if(id%32==0)
	{
		{
			time_stamp[id*2]=start;
			time_stamp[id*2+1]=end;
		}
		out[id]=q;
	}
    // __syncthreads();
    // return q;
    // printf("id=%d, start=%d, end=%d, val=%f,\n",id,start,end,(end-start)/512.0);
}

template<typename REAL>
void test() {
    // Initialize variables
    REAL p = 0.5;
    REAL q = 0.5;
    REAL *out;
    unsigned int *time_stamp;
    unsigned int num_elements = 32;
    size_t out_size = num_elements * sizeof(REAL);
    size_t time_stamp_size = num_elements * sizeof(unsigned int) *2;

    // Allocate memory on the GPU
    cudaMalloc((void**)&out, out_size);
    cudaMalloc((void**)&time_stamp, time_stamp_size);

    // Launch the kernel
    // int block_size = 256;
    // int num_blocks = (num_elements + block_size - 1) / block_size;
    kernel<REAL><<<1, 32>>>(p, q, out, time_stamp);

    // Copy the results back to the CPU
    float *out_cpu = new float[num_elements];
    unsigned int *time_stamp_cpu =(unsigned int*) malloc(time_stamp_size);
    // cudaMemcpy
    cudaMemcpy(time_stamp_cpu, time_stamp,  time_stamp_size , cudaMemcpyDeviceToHost);
    std::string str=sizeof(REAL)==sizeof(double)?"fp64":"fp32";
    printf("%s, lattency is %f\n",str.c_str(), (time_stamp_cpu[1]-time_stamp_cpu[0])/1024.0/2);
    // Print the time stamps
  
    cudaFree(out);
    cudaFree(time_stamp);

    // Free the memory on the CPU
    delete[] out_cpu;
    delete[] time_stamp_cpu;

    // return 0;
}
int main ()
{
    test<float>();
    test<double>();
    return 0;
}