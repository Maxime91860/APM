
all : tp1_cu.pgr tp1_classique.pgr

CC=nvcc

tp1_cu.pgr : tp1.cu
	$(CC) $< -g -O3 -o $@
	#nvcc tp1.cu -O3 -o tp1.pgr

tp1_classique.pgr : tp1_classique.c
	gcc $< -g -O3 -o $@

clean : 
	rm -rf tp1_cu.pgr tp1_classique.pgr