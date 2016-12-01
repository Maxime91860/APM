
all : tp1_check

tp1_check : tp1_check.cu
	nvcc -arch=sm_35 -keep -keep-dir CUDAIMG  -c $<

#-arch=sm_10 

clean : 
	rm -rf tp1_check tp1_check.o CUDAIMG