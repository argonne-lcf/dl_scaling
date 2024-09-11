from time import perf_counter_ns
import sys
t1 = perf_counter_ns() 
import intel_extension_for_pytorch  # Added Extra
import torch.nn.parallel
import horovod.torch as hvd
t2 = perf_counter_ns() 
import_timer = t2 - t1
t3 = perf_counter_ns() 
hvd.init()
t4 = perf_counter_ns() 
init_timer = t4 - t3
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

for _ in range(50):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    # print(x)
    t5 = perf_counter_ns() 
    y = hvd.allreduce(x, average=False)
    t6 = perf_counter_ns()
    elapsed1.append(t6 - t5)

if hvd.rank() == 0:
    print(import_timer)
    print(init_timer)
    for e in elapsed1:
        print(e)