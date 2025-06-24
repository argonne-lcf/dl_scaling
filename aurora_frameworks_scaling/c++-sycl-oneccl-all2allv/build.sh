#!/bin/bash

module load frameworks
mpicxx -o all2allv_ccl all2allv_ccl.cpp -fsycl -lmpi \
    -I/opt/aurora/24.347.0/oneapi/ccl/2021.14/include \
    -L/opt/aurora/24.347.0/oneapi/ccl/2021.14/lib -lccl


