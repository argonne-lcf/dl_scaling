# all_reduce

Author : Kaushik 

Date: March 1, 2024

## Time to first all reduce large and second all reduce in CCL, horovod and MPI4py

1. tensor size 1024 X 1024 - 5 loops
2. Per rank XPU results reported here. Total ppn 12. 
3. Rerunning Large node - Got the error on the blank cells - libfabric: Cassini Event Queue overflow detected. Rerunning with the patch env vars FI_CXI_*
4. In the first all_reduce MPI4py has the lowest cost, then horovod and then CCL. 
5. The first call of all_reduce is higher than the second in all frameworks. Some of the load from init has been pushed to the first call. 
6. CCL and HVD are using the baseline C MPI in the backend. Measuring them currently. 
7. With a very high one time cost with CCL, the second all_reduce has a low cost. 

estimate on : 

2500 Nodes  ~ 30 mins? 

10000 Nodes   ~ 2 hours ? 

12 ppn per node in Aurora 4 ppn per node in polaris. All values reported are per rank. 

All 128, 256 and 512 results were verified at least thrice. 

pushed from import to  init to the first all reduce call. 

Aurora - This is from /tmp local-frameworks.tar on aurora not from /soft/datascience  

polaris - conda 2023-10-04 

  Payload   x = torch.ones([1024,1024])

### First all_reduce

| Nodes | Torch CCL Aurora | Torch NCCL Polaris | HVD Aurora | HVD Polaris | MPI4py Aurora | BASELINE C MPI Aurora | CCL without torch Aurora |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.94775 | 8.21732 ( consistent) | 9.74057 | 9.20113 | 0.01930 |  |  |
| 2 | 10.03198 | 3.36613 | 9.99405 | 3.69251 | 0.02490 |  |  |
| 4 | 10.20504 | 2.98873 | 10.37532 | 2.77171 | 0.02648 |  |  |
| 8 | 10.40593 | 3.50520 | 10.79499 | 3.23226 | 0.02841 |  |  |
| 16 | 11.30797 | 4.36747 | 11.68624 | 2.98612 | 0.02708 |  |  |
| 32 | 12.12907 | 5.56126 | 16.65910 | 4.48727 | 0.02914 |  |  |
| 64 | 17.44648 | 9.13785 | 25.68926 | 6.06149 | 0.02962 |  |  |
| 128 | 32.27672 | 15.50395 | 61.12210 | 8.77237 | 0.02997 |  |  |
| 256 | 97.19558 | 28.77387 | 149.76112 | 15.06605 | 0.03006 |  |  |
| 512 | 315.81430 | 41.70092(400 nodes) | 459.43906 | 21.39870 (400 nodes) | 0.16477 |  |  |
|  |  |  |  |  |  |  |  |

### Second all_reduce

| Nodes | Torch CCL Aurora | Torch NCCL Polaris | HVD Aurora | HVD Polaris | MPI4py Aurora | BASELINE C MPI Aurora | CCL without torch Aurora |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.00121 | 0.00012 | 0.00162 | 0.18915 | 0.01127 |  |  |
| 2 | 0.00225 | 0.00754 | 0.00283 | 0.19931 | 0.01524 |  |  |
| 4 | 0.00233 | 0.00925 | 0.00305 | 0.56032 | 0.01690 |  |  |
| 8 | 0.00262 | 0.00011 | 0.00379 | 0.58930 | 0.01654 |  |  |
| 16 | 0.00237 | 0.00010 | 0.00802 | 0.50367 | 0.01673 |  |  |
| 32 | 0.00311 | 0.00212 | 0.00766 | 0.58048 | 0.01752 |  |  |
| 64 | 0.00405 | 0.00010 | 0.00860 | 0.49073 | 0.01711 |  |  |
| 128 | 0.00450 | 0.00011 | 0.10009 | 0.91565 | 0.01693 |  |  |
| 256 | 0.00683 | 0.00213 | 3.84507 | 0.81109 | 0.01720 |  |  |
| 512 | 0.01170 | 0.00212(400 nodes) | 7.67741 | 1.04472(400 nodes) | 0.01868 |  |  |

For the first all reduce, at 512 nodes Torch is 8x faster with NCCL in polaris compared to aurora. HVD is 21x faster in polaris compared to aurora.

For the second all reduce at 512 nodes Torch is 5x faster with NCCL in polaris compared to aurora. HVD is 55x faster in polaris compared to aurora.

polaris

```
dist.init_process_group(backend = "nccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=120))
dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()
print("dist_my_rank = %d  dist_world_size = %d" % (dist_my_rank, dist_world_size))
torch.cuda.set_device(dist_my_rank%4)

for _ in range(5):
    x = torch.ones([1024,1024]).cuda()
    dist.all_reduce(x,op=dist.ReduceOp.SUM)

```

Aurora

```
dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=120))
dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()
print("dist_my_rank = %d  dist_world_size = %d" % (dist_my_rank, dist_world_size))
device  = torch.device(f"xpu:{dist_my_rank%12}")

for _ in range(5):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    dist.all_reduce(x,op=dist.ReduceOp.SUM)  # Added Extra op

```

![Untitled](all_reduce%20946becedc10341b29609bba8a745de62/Untitled.png)

![Untitled](all_reduce%20946becedc10341b29609bba8a745de62/Untitled%201.png)

![Untitled](all_reduce%20946becedc10341b29609bba8a745de62/Untitled%202.png)

![Untitled](all_reduce%20946becedc10341b29609bba8a745de62/Untitled%203.png)