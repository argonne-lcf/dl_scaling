#!/bin/bash

module load conda
conda activate

cd /eagle/datascience/zhaozhenghao/workspace/methods/mnist/

# Remove hostfile if it exists
if [ -f "hostfile" ]; then
    echo "Found hostfile, removing it..."
    rm hostfile
fi

# Remove .deepspeed_env if it exists
if [ -f ".deepspeed_env" ]; then
    echo "Found .deepspeed_env, removing it..."
    rm .deepspeed_env
fi

cat $PBS_NODEFILE > hostfile
echo "hostfile created"
# sed -e 's/$/ slots=4/' -i hostfile

echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo ".deepspeed_env created"

# deepspeed --hostfile=hostfile mnist_ds.py --deepspeed --deepspeed_config mnist_ds_config.json
mpirun -ppn 4 --hostfile=hostfile python mnist_ds.py --deepspeed --deepspeed_config mnist_ds_config.json