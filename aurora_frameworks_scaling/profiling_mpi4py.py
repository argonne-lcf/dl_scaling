import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
t1 = datetime.now()
from mpi4py import MPI
t2 = datetime.now()
elapsed = (t2 - t1).total_seconds()
print(f"F-profiling complted at t2 {t2} (import mpi4py - time : {elapsed:.5f})")

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

for _ in range(50):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    # print(x)
    y = torch.empty((1024,1024))
    MPI.COMM_WORLD.Barrier()
    t5 = datetime.datetime.now() 
    comm.Allreduce(x, y, op=MPI.SUM)
    # print(y)    
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds() 
    print(f"Python: Elapsed time in each iter for all_reduce : {elapsed:.5f})")