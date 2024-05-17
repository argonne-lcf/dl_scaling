import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
import sys
import horovod.torch as hvd

hvd.init()

hvd_local_rank      = hvd.local_rank()
hvd_size            = hvd.size()
# print("hvd_local_rank = %d  hvd_size = %d" % (hvd_local_rank, hvd_size))

def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{hvd_local_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()
dim_size=int(int(sys.argv[1])/4)
elapsed1=[]

for _ in range(100):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    t4 = datetime.datetime.now() 
    y = hvd.allreduce(x, average=False)
    t5 = datetime.datetime.now()
    elapsed = (t5 - t4).total_seconds() * 10**6
    elapsed1.append(elapsed)


if hvd.rank() == 0:
    for e in elapsed1:
        print(e)