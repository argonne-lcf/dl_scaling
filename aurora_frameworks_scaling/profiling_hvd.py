import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
t1 = datetime.now()
import horovod.torch as hvd
t2 = datetime.now()
elapsed = (t2 - t1).total_seconds()
print(f"F-profiling complted at t2 {t2} (import hvd - time : {elapsed:.5f})")

t3 = datetime.now()
hvd.init()
t4 = datetime.now()
elapsed = (t4 - t3).total_seconds()
print(f"F-profiling complted at t4 {t4} (import hvd init - time : {elapsed:.5f})")



hvd_local_rank      = hvd.local_rank()
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

for _ in range(50):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    # print(x)
    t5 = datetime.datetime.now() 
    y = hvd.allreduce(x, average=False)
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds() 
    print(f"Python: Elapsed time in each iter for all_reduce : {elapsed:.5f})")
