#!/bin/bash

module load conda
conda activate

for bs in 32 64 128 256 512 1024 2048 4096 8192 16384
do
	echo "Start training at batch size $bs"
	CUDA_VISIBLE_DEVICES=0 python mnist_ori.py --batch-size $bs
done
