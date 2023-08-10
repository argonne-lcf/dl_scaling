#!/bin/sh
module load conda/2022-07-19; conda activate
export TMPDIR=./results
export MPICH_GPU_SUPPORT_ENABLED=0
export CPATH=/grand/datascience/hzheng/mlperf-2022/optimized-hpc/boost_1_80_0/:$CPATH
export LD_LIBRARY_PATH=/grand/datascience/zhaozhenghao/tools/libboost:$LD_LIBRARY_PATH

source /grand/datascience/zhaozhenghao/envs/cosmoflow/bin/activate
export PYTHONPATH=/grand/datascience/zhaozhenghao/envs/cosmoflow/lib/python3.8/site-packages/:$PYTHONPATH
