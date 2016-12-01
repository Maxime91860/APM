all : err1.pgr

err1.pgr : err1.cu
	nvcc -I /usr/local/cuda-7.5/samples/common/inc/ $< -o $@

#-arch=sm_10 

clean : 
	rm -rf *.pgr