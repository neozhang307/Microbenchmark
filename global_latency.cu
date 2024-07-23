# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int * my_array, int array_length, int iterations,  unsigned int * duration, unsigned int *index);


void parametric_measure_global(int N, int iterations, int stride);

void measure_global();


int main(){

	cudaSetDevice(0);

	measure_global();

	cudaDeviceReset();
	return 0;
}


void measure_global() {

	int N, iterations, stride; 
	//stride in element
	iterations = 10;
	
	N = 1024 * 1024* 1024/sizeof(unsigned int)*4; //in element
	stride=256;
	// for (stride = 1; stride <= N/2; stride*=2) 
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	printf("%s\t", deviceProp.name);

	{
		printf("%f GB array, \t", N/1024.0/1024/1024);
		printf("Stride = %d element, %d bytes\t", stride, stride * sizeof(unsigned int));
		parametric_measure_global(N, iterations, stride );
		// printf("===============================================\n\n");
	}
}


void parametric_measure_global(int N, int iterations, int stride) {
	cudaDeviceReset();
	int sm_count;
  	cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
	int gdim=1;
	int i;
	unsigned int * h_a;
	/* allocate arrays on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
	unsigned int * d_a;
	/* allocate arrays on GPU */
	cudaMalloc ((void **) &d_a, sizeof(unsigned int) * (N+2));

   	/* initialize array elements on CPU with pointers into d_a. */
	
	for (i = 0; i < N; i++) {		
	//original:	
		h_a[i] = (i+stride)%N;	
	}

	h_a[N] = 0;
	h_a[N+1] = 0;
	/* copy array elements from CPU to GPU */
    
	cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

	unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*256*gdim);
	unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*256*gdim);

	unsigned int *duration;
	cudaMalloc ((void **) &duration, sizeof(unsigned int)*256*gdim);

	unsigned int *d_index;
	cudaMalloc( (void **) &d_index, sizeof(unsigned int)*256*gdim );


	// cudaThreadSynchronize ();
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(gdim);

	void* KernelArgs[] ={(void**)&d_a,(void**)&N,
    	(void*)&iterations,(void*)&duration,(void*)&d_index};

	cudaEvent_t start[2], stop[2];
	cudaEventCreate(&start[0]);
	cudaEventCreate(&stop[0]);

	cudaEventRecord(start[0]);

	for(int i=0; i<10; i++)
	{
		cudaLaunchCooperativeKernel((void*)global_latency,Dg, Db,KernelArgs,0,0);
	}

	cudaEventRecord(stop[0]);
	cudaEventSynchronize(stop[0]);

	float milliseconds[0];
	cudaEventElapsedTime(&milliseconds[0], start[0], stop[0]);

	// printf("Kernel execution time: %f ms\n", milliseconds[0]);

	cudaEventCreate(&start[1]);
	cudaEventCreate(&stop[1]);

	cudaEventRecord(start[1]);
	iterations=20;
	for(int i=0; i<10; i++)
	{
		cudaLaunchCooperativeKernel((void*)global_latency,Dg, Db,KernelArgs,0,0);
	}

	cudaEventRecord(stop[1]);
	cudaEventSynchronize(stop[1]);

	cudaEventElapsedTime(&milliseconds[1], start[1], stop[1]);

	// printf("Kernel execution time: %f ms\n", milliseconds[1]);
	cudaEventDestroy(start[1]);
	cudaEventDestroy(stop[1]);



	cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int)*256*gdim, cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*256*gdim, cudaMemcpyDeviceToHost);

	// cudaThreadSynchronize ();
	// printf("");
	double elapsetime = (double)((double)milliseconds[1]/10-milliseconds[0]/10)*1000*1000/256/10;
	double elapsecycle=0;
	for(i=0;i<256;i++)
	{
		elapsecycle+=h_timeinfo[i];
		fprintf(stderr,"%d\t", h_index[i]);
		for(int g=0; g<gdim; g++)
		{
			fprintf(stderr,"%d ", h_timeinfo[i+256*g]);
		}
		fprintf(stderr,"'\n");
	}
	printf("Average latency: %f ns\t Average latency %f cycle \n", elapsetime, elapsecycle/256.0);
	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(d_index);
	cudaFree(duration);


	/*free memory on CPU */
	free(h_a);
	free(h_index);
	free(h_timeinfo);
	
	cudaDeviceReset();	

}



__global__ void global_latency (unsigned int * my_array, int array_length, int iterations, unsigned int * duration, unsigned int *index) {

	unsigned int start_time, end_time;
	unsigned int j = index[255]; 

	__shared__ unsigned int s_tvalue[256];
	__shared__ unsigned int s_index[256];

	int k;

	for(k=0; k<256; k++){
		s_index[k] = 0;
		s_tvalue[k] = 0;
	}

	cg::grid_group gg = cg::this_grid();
	
	// if(blockDim.x==0)
	// {
	// 	j = my_array[j];
	// 	s_index[k]= j;
	// 	my_array[array_length] = j;
	// }
	// gg.sync();
    for(int i=0; i<iterations; i++)
    {
        for (k = 0; k <256; k++) {
		
                start_time = clock();

                j = my_array[j];
                s_index[k]= j;
                end_time = clock();

                s_tvalue[k] = end_time-start_time;

        }
    }
	

	my_array[array_length] = j;
	my_array[array_length+1] = my_array[j];

	for(k=0; k<256; k++){
		index[k]= s_index[k];
		duration[k] = s_tvalue[k];
	}
}


