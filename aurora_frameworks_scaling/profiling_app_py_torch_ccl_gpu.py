import torch.nn.parallel
import datetime
from mpi4py import MPI
import os
import socket
import sys
import intel_extension_for_pytorch  # Added Extra
import torch.distributed as dist
import oneccl_bindings_for_pytorch


MPI.COMM_WORLD.Barrier()

os.environ['RANK']          = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE']    = str(os.environ.get('PMI_SIZE', 1))
mpi_world_size              = MPI.COMM_WORLD.Get_size()
mpi_my_rank                 = MPI.COMM_WORLD.Get_rank()

if mpi_my_rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1] 
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None

master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)

dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=120))

dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{dist_my_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()

dim_size=int(int(sys.argv[1])/4)
MPI.COMM_WORLD.Barrier()

elapsed1=[]

for _ in range(100):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    t4 = datetime.datetime.now() 
    dist.all_reduce(x, op=dist.ReduceOp.SUM)  # Added Extra op
    t5 = datetime.datetime.now()
    elapsed = (t5 - t4).total_seconds() * 10**6
    elapsed1.append(elapsed)

if mpi_my_rank == 0:
    for e in elapsed1:
        print(e)