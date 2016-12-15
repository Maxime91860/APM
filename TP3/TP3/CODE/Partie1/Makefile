
all : stream_ex1.pgr

stream_ex1.pgr : stream_ex1.cu
	nvcc $^ -o $@

exec : stream_ex1.pgr
	./$^

clean : 
	rm -rf *.pgr