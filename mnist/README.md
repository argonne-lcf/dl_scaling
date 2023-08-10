# ResNet50 with DDP, Horovod, and DeepSpeed

This repository contains the PyTorch implementations using Distributed Data Parallel (DDP), Horovod, and DeepSpeed for a CNN model on [MNIST](https://ieeexplore.ieee.org/abstract/document/6296535) dataset. The original implementation is from [PyTorch official examples](https://github.com/pytorch/examples). The implementations in this repositories are intended for usage with the [ALCF](https://alcf.anl.gov). Follow the instructions below to get started.

## 1. Conda Environment Setup
Load the Conda environment using the following commands:
```bash
# For PyTorch and Horovod
module load conda 
conda activate

# For DeepSpeed
module load conda/2023-01-10-unstable
conda activate
```

## 2. Running with Different Implementations
### PyTorch DDP
```bash
aprun -n 8 -N 4 python mnist_ddp.py --batch-size 64 
```

### Horovod
```bash
aprun -n 8 -N 4 python mnist_hvd.py --batch-size 64 
```

### DeepSpeed
```bash
mpiexec --verbose --envall -n 8 --ppn 4 --hostfile "${PBS_NODEFILE}" python mnist_ds.py --deepspeed_config scripts/deepspeed/ds_config.json
```

## Additional Information

For additional examples, refer to the [scripts](scripts/) folder. Update directories and configurations accordingly for your specific setup.