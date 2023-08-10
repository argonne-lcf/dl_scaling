#!/bin/bash

module load conda
conda activate

for n in 4
do
	for bs in 32 64 128 256 512 1024 2048
	do
		echo "Start training at batch size $bs on $n GPUs"
		aprun -n $n python mnist_ddp.py --batch-size $bs
	done
done
