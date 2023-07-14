#!/bin/bash

module load conda
conda activate

source /grand/datascience/zhaozhenghao/envs/vltvg/bin/activate

cd /grand/datascience/zhaozhenghao/workspace/methods/VLTVG
aprun -n 8 -N 4 python train_hvd.py --config configs/VLTVG_R101_referit_ddp.py --checkpoint_latest --checkpoint_best