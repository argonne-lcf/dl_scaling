#!/bin/bash

module load frameworks
mpicxx -o all2all_mpi all2all_mpi.cpp -fsycl -lmpi #-I/opt/aurora/24.086.0/oneapi/compiler/2024.1/include/sycl/


