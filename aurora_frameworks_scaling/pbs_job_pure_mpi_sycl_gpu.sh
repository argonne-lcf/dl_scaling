#!/bin/bash -x
# qsub -l nodes=100 -q workq  -l walltime=02:00:00 -l filesystems=gila -A  Aurora_deployment 
#PBS -A Aurora_deployment
#PBS -k doe

module use /soft/modulefiles
module load frameworks/2023.12.15.001 
module list

cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on nodes `cat $PBS_NODEFILE`
NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

CPU_BINDING1=list:3:4:11:12:19:20:27:28:35:36:43:44:55:56:63:64:71:72:79:80:87:88:95:96 
EXT_ENV="--env FI_CXI_DEFAULT_CQ_SIZE=1048576"

which python
mkdir -p ./out$PBS_JOBID/profiling_app_pure_mpi_sycl_gpu 

for BUF_SIZE in 1 4 16 64 256 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864 268435456
do
    date
        mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1  \
        ./profiling_app_pure_mpi_sycl_gpu ${BUF_SIZE} > ./out/profiling_app_pure_mpi_sycl_gpu/${PBS_JOBID}_${NNODES}_${NRANKS}_${RANKS_PER_NODE}_${BUF_SIZE}_c_mpi.txt
    date 
done