CC = icx 
LD = $(CC)
CFLAGS =  -O0 -g -DNDEBUG 
LDFLAGS = -fiopenmp  -fopenmp-targets=spir64_gen -Xs "-device pvc" -O0 -g

TARGET = $(file)
OBJS   =  $(file).o

all: grid

%.o : %.c
	$(CC) $(CFLAGS)  -c $<

grid: $(OBJS)
	$(LD) $(LDFLAGS) -o $(TARGET) $(OBJS) \
	/opt/aurora/23.073.0/CNDA/mpich/51.2/mpich-ofi-all-icc-default-pmix-gpu-drop51/lib/libmpi.so

clean:
	rm -f $(TARGET) $(OBJS)
	./clear
 