# init

## Time to init

### Summary:

1. Per rank results reported here. Total ppn 12. 
2. There is I/O activity happening during init. Communicator cost from communication. The location from where you init does have impact on the init time
3. Because of the I/O activity during init, it is good to consider DAOS here. 
4. TMP has a lower time to init all 3 framrworks
5. MPI4py does not have init time in python. 
6. As before the I/O activity in /slash soft init is higher than /lus init 
7. HVD has a slightly ( Roughly ) higher init time than CCL. This is due to the I/O activity and not a problem of MPI. 
8. The init issue we are dealing here is currently in few seconds. 
9. For now the init issue can also be resolved with the /tmp method. Need to verify beyond 512 nodes. Waiting on PING issue. 
10. 

Todo : Measure just the one ccl init with the torch CCL init. One CCL might have less time on HVD. 

used python bindings.

Measure strace on the init phase - openat cost etc. on just single node. 

| Nodes | CCL slash soft init | HVD slash soft init | MPI4py slash soft init |  | CCL lustre init | HVD lustre init | MPI4py lustre init |  | CCL tmp INIT | HVD tmp INIT | MPI4py tmp INIT |  | CCL DAOS INIT | HVD DAOS INIT | MPI4PY DAOS INIT |  | Baseline just one CCL init without torch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.03577 | 3.65892 |  |  | 1.88339 | 4.05971 |  |  | 0.0456 | 3.79462 |  |  |  |  |  |  |  |
| 2 | 0.02726 | 6.71537 |  |  | 5.02328 | 3.38675 |  |  | 0.0132 | 3.22028 |  |  |  |  |  |  |  |
| 4 | 5.04836 | 9.30146 |  |  | 5.0367 | 48.06468 |  |  | 5.01723 | 3.40178 |  |  |  |  |  |  |  |
| 8 | 5.05042 | 22.36316 |  |  | 5.0644 | 3.70899 |  |  | 0.06308 | 3.51047 |  |  |  |  |  |  |  |
| 16 | 0.11111 | 6.32664 |  |  | 0.06504 | 171.88087 |  |  | 0.07338 | 3.66009 |  |  |  |  |  |  |  |
| 32 | 5.1001 | 19.61483 |  |  | 5.05809 | 9.38179 |  |  | 0.10803 | 3.66918 |  |  |  |  |  |  |  |
| 64 | 5.13289 | 11.10524 |  |  | 5.07404 | 34.83611 |  |  | 5.0606 | 3.76971 |  |  |  |  |  |  |  |
| 128 | 0.40757 | 16.40086 |  |  | 5.05912 | 38.44712 |  |  | 6.05937 | 3.74361 |  |  |  |  |  |  |  |
| 256 | 0.89848 | 64.08288 |  |  | 6.1039 | 34.27702 |  |  | 5.20578 | 4.15317 |  |  |  |  |  |  |  |
| 512 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Add 256 and 512 results here.