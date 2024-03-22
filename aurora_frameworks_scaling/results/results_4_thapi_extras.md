# extras

## Thapi L0 profiling for all reduce and mem consumption

CCL BACKEND_ZE | 1 Hostnames | 36 Processes | 57 Threads |

HVD BACKEND_ZE | 1 Hostnames | 36 Processes | 72 Threads |                   ( creates more threads)

MPI4py BACKEND_ZE | 1 Hostnames | 36 Processes | 36 Threads |

CCL Device profiling | 1 Hostnames | 12 Processes | 24 Threads | 6 Devices | 12 Subdevices |

HVD Device profiling | 1 Hostnames | 12 Processes | 36 Threads | 6 Devices | 12 Subdevices |

MPI4py Device profiling | 1 Hostnames | 12 Processes | 12 Threads | 6 Devices | 13 Subdevices |

Part 1 : CPU Host profiling Total Time and Total Number of calls

Part 2: GPU profiling  Total Time and Total Number of calls

Part 3: Memory profiling Total memory and  Total Number of calls

|  |  |   Total CPU Time | Total CPU calls |  |  |   Total CPU Time | Total CPU calls |  |  |   Total CPU Time | Total CPU calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 
  
  1 | CCL | 1.85min | 350753 |  | HVD | 2.09min | 319468 |  | MPI4py | 3.12s | 907154 |
| 2 | CCL | 304.39ms | 11400 |  | HVD | 292.52ms | 12000 |  | MPI4py | 1.24s | 3000 |
| 4 | CCL | 1.85min | 387095 |  | HVD | 1.86min | 339299 |  | MPI4py | 2.97s | 862776 |
| 8 | CCL | 1.87min | 393338 |  | HVD | 1.85min | 347886 |  | MPI4py | 2.85s | 827504 |
| 16 | CCL | 1.85min | 389583 |  | HVD | 1.89min | 350250 |  | MPI4py | 2.97s | 862586 |
| 32 | CCL | 1.90min | 390379 |  | HVD | 1.87min | 348553 |  | MPI4py | 2.88s | 852339 |
| 64 | CCL | 1.85min | 384236 |  | HVD | 1.84min | 145796 |  | MPI4py | 2.92s | 821632 |
| 128 | CCL | 1.97min | 383091 |  | HVD | 1.54s | 64255 |  | MPI4py | 3.39s | 861671 |
| 256 | CCL | 2.01min | 389735 |  | HVD | 1.16s | 64257 |  | MPI4py | 3.13s | 845094 |
| 512 | CCL |  |  |  | HVD |  |  |  | MPI4py | 3.35s | 808502 |
|  |  |   Total GPU Time | Total GPU calls |  |  |   Total GPU Time | Total GPU calls |  |  |   Total GPU Time | Total GPU calls |
| 1 | CCL | 300.79ms | 10200 |  | HVD | 284.08ms | 10800 |  | MPI4py | 1.28s | 3000 |
| 2 | CCL | 1.86min | 391994 |  | HVD | 2.14min | 331795 |  | MPI4py | 3.06s | 894462 |
| 4 | CCL | 302.96ms | 11400 |  | HVD | 304.27ms | 12000 |  | MPI4py | 1.06s | 3000 |
| 8 | CCL | 306.18ms | 11400 |  | HVD | 294.59ms | 12000 |  | MPI4py | 878.28ms | 3000 |
| 16 | CCL | 305.74ms | 11400 |  | HVD | 298.81ms | 12000 |  | MPI4py | 1.06s | 3000 |
| 32 | CCL | 311.04ms | 11400 |  | HVD | 300.59ms | 12000 |  | MPI4py | 1.02s | 3000 |
| 64 | CCL | 296.16ms | 11400 |  | HVD | 82.84ms | 4308 |  | MPI4py | 858.53ms | 3000 |
| 128 | CCL | 297.18ms | 11400 |  | HVD |  |  |  | MPI4py | 1.05s | 3000 |
| 256 | CCL | 301.18ms | 11400 |  | HVD |  |  |  | MPI4py | 981.47ms | 3000 |
| 512 | CCL |  |  |  | HVD |  |  |  | MPI4py | 753.75ms | 3000 |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |   Total Mem Bytes | Total Mem calls |  |  |   Total Mem Bytes | Total Mem calls |  |  |   Total Mem Bytes | Total Mem calls |
| 1 | CCL | 6.63GB | 7826 |  | HVD | 6.63GB | 7826 |  | MPI4py | 5.58GB | 3026 |
| 2 | CCL | 7.05GB | 9026 |  | HVD | 7.05GB | 9026 |  | MPI4py | 5.58GB | 3026 |
| 4 | CCL | 7.05GB | 9026 |  | HVD | 7.05GB | 9026 |  | MPI4py | 5.58GB | 3026 |
| 8 | CCL | 7.05GB | 9026 |  | HVD | 7.05GB | 9026 |  | MPI4py | 5.58GB | 3026 |
| 16 | CCL | 7.05GB | 9026 |  | HVD | 7.05GB | 9026 |  | MPI4py | 5.58GB | 3026 |
| 32 | CCL | 7.05GB | 9026 |  | HVD | 7.05GB | 9026 |  | MPI4py | 5.58GB | 3026 |
| 64 | CCL | 7.05GB | 9026 |  | HVD | 3.07GB | 3458 |  | MPI4py | 5.58GB | 3026 |
| 128 | CCL | 7.05GB | 9026 |  | HVD | 595.59MB | 38 |  | MPI4py | 5.58GB | 3026 |
| 256 | CCL | 7.05GB | 9026 |  | HVD | 595.59MB | 38 |  | MPI4py | 5.58GB | 3026 |
| 512 | CCL |  |  |  | HVD | 1.04s | 63372 |  | MPI4py | 5.58GB | 3026 |
|  |  |  |  |  |  |  |  |  |  |  |  |