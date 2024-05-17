import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
import sys
t0 = datetime.datetime.now()

from mpi4py import MPI

t1 = datetime.datetime.now()
elapsed = (t1 - t0).total_seconds() 
print(f"import mpi4py time : {elapsed:.5f}")

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_local_rank = comm.Get_rank()
print("mpi_local_rank = %d  mpi_size = %d" % (mpi_local_rank, mpi_size))

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{mpi_local_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()
print(device)
# torch.xpu.set_device(mpi_local_rank if torch.xpu.device_count() > 1 else 0)
dim_size=int(int(sys.argv[1])/4)
MPI.COMM_WORLD.Barrier()
    
for _ in range(50):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    y = torch.empty((1,dim_size),dtype=torch.float32)
    t4 = datetime.datetime.now() 
    comm.Allreduce(x, y, op=MPI.SUM)
    t5 = datetime.datetime.now()
    # print(y)
    elapsed = (t5 - t4).total_seconds() * 10**6
    if mpi_local_rank == 0:
        print(f"Python: Elapsed time in each iter for all_reduce : {elapsed:.5f} u.sec u.sec for bytes {int(sys.argv[1])} by rank {mpi_local_rank}")
