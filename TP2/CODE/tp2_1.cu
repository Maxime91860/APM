#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define SIZE 102400
#define MOD 102399
#define STEP 128

/* ARRAY A INITIALIZER */
void init_a(int * a)
{
    int i;
    for(i=0; i<SIZE; i++)
    {
        a[i] = 1;
    }
}

/* ARRAY B INITIALIZER */
void init_b(int * b)
{
	int i, j;

	j=0;

	for(i=0; i<SIZE-1; i++)
	{
		b[j] = i;
		j = (j+STEP)%MOD;
	}	

    b[SIZE-1] = SIZE-1;
}

/* CHECKING A VALUES */
int check_a(int * a)
{
    int i;
    int correct = 1;
	for(i=0; i<SIZE; i++)
	{
		if(a[i] != (i+1)) 
		{
			correct = 0;
		} 
	}	

    return correct;
}


/* CUDA FUNCTION */
__global__ void mykernel(int * a, int * b, int N)
{
	/* A COMPLETER */
	// int 
	int j;
	for(j=0; j<100; j++){
		int i = threadIdx.x + j*blockDim.x;
		if(i < N){
			// printf("a[%d] = %d, b[%d] = %d \n",i,a[i], i, b[i] );
			a[ b[i] ] += b[i];
		}
	}
	

}


int main(int argc, char * argv[])
{

	int taille = sizeof(int)*SIZE;

	int * h_a = (int *)malloc(taille);
	int * h_b = (int *)malloc(taille);

    init_a(h_a);
	init_b(h_b);

	int* d_a;
	int* d_b;

	/* A COMPLETER */    
	cudaMalloc((void**)&d_a, taille);
	cudaMalloc((void**)&d_b, taille);

	cudaMemcpy (d_a, h_a, taille, cudaMemcpyHostToDevice);
	cudaMemcpy (d_b, h_b, taille, cudaMemcpyHostToDevice);


	dim3 nBlocks;
	dim3 nThperBlock;

	nBlocks.x = 1;
	
	nThperBlock.x = 1024;

	mykernel<<< nBlocks , nThperBlock >>>(d_a, d_b, SIZE);

	
	/* INSERT CUDA COPY HERE */
    /* A COMPLETER */
    cudaMemcpy (h_a, d_a, taille, cudaMemcpyDeviceToHost);



	int correct = check_a(h_a);;
	
	if(0 == correct)
	{
		printf("\n\n ******************** \n ***/!\\ ERROR /!\\ *** \n ******************** \n\n");
	}
	else
	{
		printf("\n\n ******************** \n ***** SUCCESS! ***** \n ******************** \n\n");
	}

	free(h_a);
	free(h_b);

	cudaFree(d_a);
	cudaFree(d_b);


	return 1;
}
