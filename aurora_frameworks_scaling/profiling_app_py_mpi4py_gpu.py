from time import perf_counter_ns
import sys


t1 = perf_counter_ns() 
import intel_extension_for_pytorch  # Added Extra
import torch.nn.parallel
from mpi4py import MPI
t2 = perf_counter_ns() 
import_timer = t2 - t1

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_local_rank = comm.Get_rank()
# print("mpi_local_rank = %d  mpi_size = %d" % (mpi_local_rank, mpi_size))

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{mpi_local_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()
# print(device)
dim_size=int(int(sys.argv[1])/4)
MPI.COMM_WORLD.Barrier()

elapsed1=[]

for _ in range(50):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    y = torch.empty((1,dim_size),dtype=torch.float32)
    t5 = perf_counter_ns() 
    comm.Allreduce(x, y, op=MPI.SUM)
    MPI.COMM_WORLD.Barrier()
    t6 = perf_counter_ns() 
    # print(y)
    elapsed1.append(t6 - t5)

if comm.Get_rank() == 0:
    print(import_timer)

    for e in elapsed1:
        print(e)