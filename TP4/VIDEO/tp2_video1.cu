#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "libvideo.h"

#define SEUIL 50
#define RED 0
#define GREEN 1
#define BLUE 2





__global__ void kernel_grey(char * frame1, int width, int length)
{	

	// ------ Version 1 seul bloc de height threads ------
	// int i = threadIdx.x; //Indice global
	// int j;
	// int line_size = width*3;
	// for(j=0; j<width; j++){
	// 	int red   = frame1[i*line_size+3*j+RED];
	// 	int green = frame1[i*line_size+3*j+GREEN];
	// 	int blue  = frame1[i*line_size+3*j+BLUE];

	// 	int moy = red/4 + (green*3/4);

	// 	frame1[i*line_size+3*j+RED] = (char)moy;
	// 	frame1[i*line_size+3*j+GREEN] = (char)moy;
	// 	frame1[i*line_size+3*j+BLUE] = (char)moy;
	// }

	// ------ Version de 16 blocs * 1024 threads ---- 
	int i = threadIdx.x + blockIdx.x * blockDim.x ;
	int nb_threads = gridDim.x * blockDim.x;
	int line_size = width*3;
	int j;

	
	for(j=i; j<length; j=j+nb_threads){
		// if(i*j < length){
			int red   = frame1[j*3+RED];
			int green = frame1[j*3+GREEN];
			// int blue  = frame1[i+j*nb_threads+BLUE];

			int moy = red/4 + (green*3/4);

			frame1[j*3+RED] = (char)moy;
			frame1[j*3+GREEN] = (char)moy;
			frame1[j*3+BLUE] = (char)moy;
		// }
	}
}



int main (int argc, char * argv[])
{
	// int i, j, k;
	int cpt_frame;
	// int cpt = 0;
	int frame_count;
	int width, height;



	printf("Opening videos - read and write\n"); fflush(stdout);

	OpenReadAndWriteVideo("./Wildlife.wmv", "./my_copy2.wmv");


	printf("----------------------------------------\n");
	frame_count = getFrameCount();
	width = getWidth();
	height = getHeight();
	printf("Frame count = %d\n", frame_count); fflush(stdout); 

	printf("Width  of frames: %d\n", width); fflush(stdout);
	printf("Height of frames: %d\n", height); fflush(stdout);


//	char * frames = (char *) malloc( sizeof(char) * frame_count * width * height * 3);
	char * frame1 = (char *) malloc( sizeof(char) * width * height * 3);
	char * frame1_device; 
	cudaMalloc((void**)&frame1_device, sizeof(char) * width * height * 3);

	// int line_size = width*3;

	for(cpt_frame = 0; cpt_frame < 100 && cpt_frame < frame_count; cpt_frame ++)
	{	

		printf("%d - Read frame with index\n", cpt_frame); fflush(stdout);
		readFrame_with_index(frame1, cpt_frame);


		if(cpt_frame > 10 && cpt_frame < 100)
		{
			cudaMemcpy(frame1_device, frame1, sizeof(char)*width*height*3, cudaMemcpyHostToDevice);
			printf("%d - GREY\n", cpt_frame); fflush(stdout);

	        /* COLOR -> GREY */
	        // kernel_grey<<<1 , height>>>(frame1_device, width, height);
	        kernel_grey<<<16 , 64>>>(frame1_device, width, height*width);

	        cudaMemcpy(frame1, frame1_device, sizeof(char)*width*height*3, cudaMemcpyDeviceToHost);

		}
		writeFrame (frame1);

	}
	printf("ECRITURE VIDEO FINIE\n");

	free(frame1);
	cudaFree(frame1_device);



	return 0;

}
