import datetime
from time import perf_counter_ns
import sys

import tensorflow as tf
import horovod.tensorflow as hvd
import intel_extension_for_tensorflow as itex
print(itex.__version__)
hvd.init()

hvd_local_rank      = hvd.local_rank()
hvd_size            = hvd.size()
print("hvd_local_rank = %d  hvd_size = %d" % (hvd_local_rank, hvd_size))

xpus = tf.config.experimental.list_physical_devices('XPU')
logical_gpus = tf.config.experimental.set_visible_devices(xpus[hvd.local_rank()], 'XPU')
print(xpus)
tf.debugging.set_log_device_placement(True)


dim_size=int(int(sys.argv[1])/4)
elapsed1=[]

for _ in range(5):
    with tf.device(f"XPU:{hvd_local_rank%12}"):
        x = tf.ones([1, dim_size],dtype=tf.float32)
        # print(x)
        t5 = perf_counter_ns() 
        y = hvd.allreduce(x, average=False)
        t6 = perf_counter_ns()
        elapsed1.append(t6 - t5)

if hvd.rank() == 0:
    for e in elapsed1:
        print(e)