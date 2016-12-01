all: tp2_video1.pgr

CC=gcc
CFLAGS= -I./libvideo/ -I.
LDFLAGS= -L./libvideo/ -lvideo -lopencv_highgui -lopencv_core -lm




%.pgr: %.o
	$(CC) $(CFLAGS) $< -o $@  $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS)  $< -c -o $@


clean:
	rm -rf *.o *.pgr
