#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=01:00:00
#PBS -l nodes=2:ppn=4
#PBS -A datascience
#PBS -l filesystems=home:grand
cd $PBS_O_WORKDIR

source ./setup.sh

aprun -n 8 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_tfr_2