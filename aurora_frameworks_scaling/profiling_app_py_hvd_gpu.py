import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
import sys
t0 = datetime.datetime.now()

import horovod.torch as hvd
t1 = datetime.datetime.now()
elapsed = (t1 - t0).total_seconds() 
print(f"import horovod.torch as hvd time : {elapsed:.5f}")

t2 = datetime.datetime.now()
hvd.init()
t3 = datetime.datetime.now()
elapsed = (t3 - t2).total_seconds() 
print(f"HVD init time : {elapsed:.5f}")

hvd_local_rank      = hvd.local_rank()
hvd_my_rank         = hvd.rank()
hvd_size            = hvd.size()
print("hvd_local_rank = %d  hvd_size = %d" % (hvd_local_rank, hvd_size))

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{hvd_local_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()
print(device)
# torch.xpu.set_device(hvd_local_rank if torch.xpu.device_count() > 1 else 0)
dim_size=int(int(sys.argv[1])/4)

for _ in range(50):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    t4 = datetime.datetime.now() 
    y = hvd.allreduce(x, average=False)
    t5 = datetime.datetime.now()
    elapsed = (t5 - t4).total_seconds() * 10**6
    if hvd_my_rank == 0:
        print(f"Python: Elapsed time in each iter for all_reduce : {elapsed:.5f} u.sec u.sec for bytes {int(sys.argv[1])} by rank {hvd_local_rank}")
