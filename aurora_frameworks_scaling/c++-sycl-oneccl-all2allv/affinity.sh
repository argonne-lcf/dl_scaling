#!/bin/bash
num_gpus=12
gpu_id=$((PALS_LOCAL_RANKID % num_gpus ))
export ZE_AFFINITY_MASK=$gpu_id
#echo $PALS_LOCAL_RANKID running on GPU $gpu_id
exec "$@"
