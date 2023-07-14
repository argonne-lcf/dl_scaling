#!/bin/bash

module load conda/2023-01-10-unstable
conda activate

source /grand/datascience/zhaozhenghao/envs/pretrained_bert/bin/activate

cd /grand/datascience/zhaozhenghao/workspace/methods/VLTVG
mpiexec --verbose --envall -n 8 --ppn 4 --hostfile "${PBS_NODEFILE}" python train_ds.py --config configs/VLTVG_R101_referit_ddp.py --polaris_nodes 2 --checkpoint_latest --checkpoint_best --deepspeed_config scripts/deepspeed/ds_8_config.json