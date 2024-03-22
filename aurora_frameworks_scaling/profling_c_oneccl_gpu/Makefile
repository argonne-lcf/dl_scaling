CXX = g++
LD = $(CXX)
CFLAGS = -O2 -g -DNDEBUG 
LDFLAGS = -O2 -g

TARGET = cpu_allreduce_test
OBJS   =  cpu_allreduce_test.o

all: grid

%.o : %.c
	$(CC) $(CFLAGS) -c $<

grid: $(OBJS)
	$(LD) $(LDFLAGS) -o $(TARGET) $(OBJS) \
	/soft/compilers/oneapi/2023.12.15.001/oneapi/ccl/2021.11/lib/libccl.so.1 \
	/soft/restricted/CNDA/updates/2023.12.15.001/oneapi/compiler/2024.0/lib/libsycl.so.7.0.0

clean:
	rm -f $(TARGET) $(OBJS)
	./clear
 