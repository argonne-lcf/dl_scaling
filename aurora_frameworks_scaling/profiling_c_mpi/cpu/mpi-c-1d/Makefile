CC = icx 
LD = $(CC)
CFLAGS = -O2 -g -DNDEBUG 
LDFLAGS = -O2 -g


TARGET = $(file)
OBJS   =  $(file).o

all: grid

%.o : %.c
	$(CC) $(CFLAGS) -c $<

grid: $(OBJS)
	$(LD) $(LDFLAGS) -o $(TARGET) $(OBJS) \
	/soft/restricted/CNDA/updates/mpich/52.2/mpich-ofi-all-icc-default-pmix-gpu-drop52/lib/libmpi.so

clean:
	rm -f $(TARGET) $(OBJS)
	./clear
 