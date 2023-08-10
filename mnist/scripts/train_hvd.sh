#!/bin/bash

module load conda
conda activate

for n in 4 8
do
	for bs in 32 64 128 256 512 1024 2048
	do
		echo "Start training at batch size $bs on $n GPUs on $((n/2)) nodes"
		aprun -n $n -N $((n/2)) python mnist_hvd.py --batch-size $bs
	done
done
