# imports

## Number of strace calls to import Total, openat() and stats() from  - lustre local frameworks

Summary:  

1. The below value is roughly a constant for 1 rank. Multiply it for N nodes and R ranks. 

```python
Measured just a single line of 

import torch.distributed as dist
or 
import horovod.torch as hvd
or
from mpi4py import MPI
```

| Total Torch CCL | 145,367 | openat Torch CCL | 25,084 | stats Torch CCL | 47,659 |
| --- | --- | --- | --- | --- | --- |
| Total Hvd | 69,371 | openat Hvd | 5,151 | stats Hvd | 43,181 |
| Total MPI4py | 145,105 | openat MPI4py | 24,975 | stats MPI4py | 47,608 |
|  |  |  |  |  |  |

For example, currently loading importing intel-torch from 1 node 1 rank makes 145367 calls. This scales linearly.

For 64 (ppn)*2048(nodes)*145367(calls) would be **19,053,543,424 calls to lustre.**

Note lustre max OPS roughly 600K

Summary: 

1. The time listed below are for 1 rank. Total PPN 12. The same constant time for all 12 ranks. Scales linearly. 
2. MPI import includes init time.
3. /lus means having the local frameworks extracted in a project dir in gecko. 
4. Eventhough /soft and lus are from the same backend, /soft is 4 X slower than /lus
5. The import issue can be currently solved with /tmp method for larger nodes. 
6. Currently stuck with the ping issue beyond > 512 nodes. 
7. The benefit of /temp method starts to kick only beyond 512 nodes. If < 512 nodes its better to use from local frameworks from a project dir
8. Compared to the init issue which is in few seconds, the import issue is in few minutes.

| Nodes | soft-CCL | soft-HVD | soft-mpi4py |  | Nodes | lus-CCL | lus-HVD | lus-mpi4py |  | Nodes | DAOS-SX-CCL | DAOS-SX-HVD | DAOS-SX-MPI4PY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 23.76304 | 25.84276 | 7.84032 |  | 1 | 16.91329 | 17.72262 | 5.54276 |  | 1 | Pending |  |  |
| 2 | 12.43426 | 12.35091 | 9.08423 |  | 2 | 10.95631 | 3.96761 | 2.73401 |  | 2 |  |  |  |
| 4 | 19.00911 | 10.72862 | 8.73533 |  | 4 | 3.60226 | 15.4931 | 2.90424 |  | 4 |  |  |  |
| 8 | 9.82093 | 15.33458 | 11.0752 |  | 8 | 2.52667 | 12.67131 | 2.91102 |  | 8 |  |  |  |
| 16 | 18.52252 | 20.58837 | 8.88195 |  | 16 | 11.08034 | 3.30889 | 2.79047 |  | 16 |  |  |  |
| 32 | 45.95038 | 36.75937 | 8.58397 |  | 32 | 12.22699 | 3.30789 | 3.04978 |  | 32 |  |  |  |
| 64 | 79.40159 | 89.44833 | 10.37787 |  | 64 | 5.91391 | 15.36768 | 3.50686 |  | 64 |  |  |  |
| 128 | 153.83436 | 159.68366 | 20.24718 |  | 128 | 3.37653 | 47.44398 | 7.2994 |  | 128 |  |  |  |
| 256 | 576.35857 | 389.59318 | 32.53079 |  | 256 | 19.65773 | 79.98923 | 15.40001 |  | 256 |  |  |  |
| 512 | 605.25613 | 603.68027 | 95.82737 |  | 512 | 8.24698 | 85.24325 | 20.26439 |  | 512 |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method | TMP Method |  |  |
| Nodes | tmp tranfer | tmp-CCL | Total TMP- CCL |  | tmp tranfer | tmp-HVD | TOTAL -TMP -HVD |  | tmp tranfer | tmp-mpi4py | Total TMP MPI4PY |  |  |
| 1 | 260 | 0.85047 | 260.85047 |  | 260 | 0.96533 | 260.96533 |  | 260 | 2.64186 | 262.64186 |  |  |
| 2 | 260 | 0.8716 | 260.8716 |  | 260 | 0.93277 | 260.93277 |  | 260 | 2.68266 | 262.68266 |  |  |
| 4 | 260 | 0.8871 | 260.8871 |  | 260 | 0.94378 | 260.94378 |  | 260 | 2.64645 | 262.64645 |  |  |
| 8 | 260 | 0.84991 | 260.84991 |  | 260 | 0.93416 | 260.93416 |  | 260 | 2.79356 | 262.79356 |  |  |
| 16 | 260 | 0.86597 | 260.86597 |  | 260 | 0.93822 | 260.93822 |  | 260 | 2.78354 | 262.78354 |  |  |
| 32 | 260 | 0.88164 | 260.88164 |  | 260 | 0.93331 | 260.93331 |  | 260 | 2.86852 | 262.86852 |  |  |
| 64 | 260 | 0.86922 | 260.86922 |  | 260 | 0.92284 | 260.92284 |  | 260 | 2.95718 | 262.95718 |  |  |
| 128 | 260 | 0.84241 | 260.84241 |  | 260 | 0.89717 | 260.89717 |  | 260 | 2.96501 | 262.96501 |  |  |
| 256 | 260 | 0.84077 | 260.84077 |  | 260 | 0.94813 | 260.94813 |  | 260 | 2.90351 | 262.90351 |  |  |
| 512 | 300 | 0.83391 | 300.83391 |  | 300 | 0.96462 | 300.96462 |  | 260 | 3.45906 | 263.45906 |  |  |

one CCL issue in > 256 nodes 

Canvas in ds_nda KV store in one CCL 

High STDEV in large nodes - 

## Time to full set of import

```python
import torch.nn.parallel
import intel_extension_for_pytorch  # Added Extra
import datetime
import oneccl_bindings_for_pytorch
import os
import socket
from mpi4py import MPI
import torch.distributed as dist
```

| Nodes |  slash soft ccl | slash soft hvd | MPI4py slash soft init |  | lus  ccl | lus hvd | MPI4py lustre init |  |  | tmp tranfer | tmp  ccl | tmp hvd | MPI4py tmp INIT |  | DAOS-SX-CCL | DAOS-SX-HVD | DAOS-SX-MPI4PY |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  | 44.88728 |  |  |  | 45.70461 |  |  | 260 |  |  | 5.24374 |  |  |  |  |  |  |
| 2 | 51.36861 | 17.57733 | 64.45655 |  | 154.32752 | 7.53602 | 45.70461 |  |  | 260 | 5.01381 | 2.41797 | 4.83445 |  |  |  |  |  |  |
| 4 | 55.86688 | 21.99815 | 44.88728 |  | 50.11577 | 5.68276 | 44.19387 |  |  | 260 | 5.01556 | 2.43541 | 5.18642 |  |  |  |  |  |  |
| 8 | 53.08752 | 177.69967 | 49.28202 |  | 46.60836 | 5.65113 | 47.68961 |  |  | 260 | 5.18202 | 3.93436 | 5.14455 |  |  |  |  |  |  |
| 16 | 65.9004 | 94.96548 | 67.04855 |  | 46.57659 | 7.24566 | 57.07946 |  |  | 260 | 5.36895 | 2.47179 | 5.0717 |  |  |  |  |  |  |
| 32 | 80.01476 | 81.82024 | 147.6838 |  | 43.99667 | 38.83774 | 48.59356 |  |  | 260 | 5.14679 | 2.44414 | 5.43864 |  |  |  |  |  |  |
| 64 | 141.3393 | 130.00782 | 158.37408 |  | 150.44628 | 20.88956 | 60.71773 |  |  | 260 | 5.1985 | 2.42901 | 5.18209 |  |  |  |  |  |  |
| 128 | 257.02557 | 265.41219 | 378.93695 |  | 64.514 | 31.20652 | 86.72031 |  |  | 260 | 5.39474 | 2.43667 | 5.3518 |  |  |  |  |  |  |
| 256 | 462.9181 | 412.89474 | 503.58247 |  | 108.7893 | 67.65972 | 188.76655 |  |  | 260 | 5.64393 | 2.64013 | 5.53286 |  |  |  |  |  |  |
| 512 |  |  |  |  |  |  | 262.9439 |  |  | 260 |  |  |  |  |  |  |  |  |  |

This table includes the below along with importing the framework

```python
import torch
import intel_extension_for_pytorch
```

| Nodes | Torch CCL import time /lus | Torch CCL import time /soft |  |  | Torch CCL import time /tmp |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | lus | soft |  | tmp tranfer costant for all ranks | tmp |  |  | daos |
| 1 | 9.30729 | 23.2586 |  | 260 | 4.95412 |  |  |  |
| 2 | 154.32752 | 51.36861 |  | 260 | 5.01381 |  |  |  |
| 4 | 50.11577 | 55.86688 |  | 260 | 5.01556 |  |  |  |
| 8 | 46.60836 | 53.08752 |  | 260 | 5.18202 |  |  |  |
| 16 | 46.57659 | 65.9004 |  | 260 | 5.36895 |  |  |  |
| 32 | 43.99667 | 80.01476 |  | 260 | 5.14679 |  |  |  |
| 64 | 150.44628 | 141.3393 |  | 260 | 5.1985 |  |  |  |
| 128 | 64.514 | 257.02557 |  | 260 | 5.39474 |  |  |  |
| 256 | 108.7893 | 462.9181 |  | 260 | 5.64393 |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  | hvd import time /lus | hvd import time /soft |  |  | hvd import time /tmp |  |  |  |
| Nodes | lus | soft |  | tmp tranfer | tmp |  |  |  |
| 1 | 5.86334 | 18.81337 |  | 260 | 2.44307 |  |  |  |
| 2 | 7.53602 | 17.57733 |  | 260 | 2.41797 |  |  |  |
| 4 | 5.68276 | 21.99815 |  | 260 | 2.43541 |  |  |  |
| 8 | 5.65113 | 177.69967 |  | 260 | 3.93436 |  |  |  |
| 16 | 7.24566 | 94.96548 |  | 260 | 2.47179 |  |  |  |
| 32 | 38.83774 | 81.82024 |  | 260 | 2.44414 |  |  |  |
| 64 | 20.88956 | 130.00782 |  | 260 | 2.42901 |  |  |  |
| 128 | 31.20652 | 265.41219 |  | 260 | 2.43667 |  |  |  |
| 256 | 67.65972 | 412.89474 |  | 260 | 2.64013 |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
|  | mpi4py import time /lus | mpi4py import time /soft |  |  | mpi4py import time /tmp |  |  |  |
| Nodes | lus | soft |  | tmp tranfer | tmp |  |  |  |
| 1 | 11.73358 | 22.77695 |  | 260 | 5.24374 |  |  |  |
| 2 | 45.70461 | 64.45655 |  | 260 | 4.83445 |  |  |  |
| 4 | 44.19387 | 44.88728 |  | 260 | 5.18642 |  |  |  |
| 8 | 47.68961 | 49.28202 |  | 260 | 5.14455 |  |  |  |
| 16 | 57.07946 | 67.04855 |  | 260 | 5.0717 |  |  |  |
| 32 | 48.59356 | 147.6838 |  | 260 | 5.43864 |  |  |  |
| 64 | 60.71773 | 158.37408 |  | 260 | 5.18209 |  |  |  |
| 128 | 86.72031 | 378.93695 |  | 260 | 5.3518 |  |  |  |
| 256 | 188.76655 | 503.58247 |  | 260 | 5.53286 |  |  |  |
| 512 | 262.9439 |  |  |  |  |  |  |  |

Tips for > 512 nodes 

pdsh 

mpiutils

aprun 

move files nathen