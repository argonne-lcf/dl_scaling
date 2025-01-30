#!/bin/bash

module load frameworks
mpicxx -o all2allv_ccl all2allv_ccl.cpp -fsycl -lmpi \
    -I/opt/aurora/24.180.3/updates/oneapi/ccl/2021.13.1_20240808.145507/include \
    -L/opt/aurora/24.180.3/updates/oneapi/ccl/2021.13.1_20240808.145507/lib -lccl


