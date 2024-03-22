#!/bin/sh -x
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l daos=default
#PBS -A Aurora_deployment
#PBS -q alcf_daos_cn
#PBS -k doe 

#PBS -o  provide_path
#PBS -e  provide_path



# #qsub -l select=1 -l place=scatter -l walltime=00:60:00 -A Aurora_deployment   -q EarlyAppAccess -k doe 





module use /soft/modulefiles
module load frameworks
module load spack-pe-gcc  thapi
module load  tools/xpu-smi

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=6          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"
export MPIR_CVAR_ENABLE_GPU=1 
# export FI_CXI_RX_MATCH_MODE=software
# export FI_CXI_DEFAULT_CQ_SIZE=1048576
# export MPIR_CVAR_CH4_MT_MODEL=lockless

# mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose  ./base_mpi_gpu_coll
# mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose  gpu_tile_compact.sh  ./base_mpi_gpu_coll
iprof  --max-name-size -1 mpiexec --env FI_CXI_DEFAULT_CQ_SIZE=16384  --env FI_CXI_OVFLOW_BUF_SIZE=8388608 --env FI_CXI_CQ_FILL_PERCENT=20 --np ${NRANKS} -ppn ${RANKS_PER_NODE} gpu_tile_compact.sh  ./base_mpi_gpu_coll
