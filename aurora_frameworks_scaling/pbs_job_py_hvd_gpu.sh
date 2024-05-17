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
mkdir -p ./out$PBS_JOBID/profiling_app_py_hvd_gpu

# Note for each BUF_SIZE, the dimension of the tensor will be BUF_SIZE/2, BUF_SIZE/2, resulting true all reduce buffer will be BUF_SIZE/2 * BUF_SIZE/2. 
# For example, BUF_SIZE=8, dim1=4 dim2=4, all reduce buffer=4*4=16 

for BUF_SIZE in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768  
do
    date 
        mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1  \
        python3 ./profiling_app_py_hvd_gpu.py  ${BUF_SIZE} \
        > ./out$PBS_JOBID/profiling_app_py_hvd_gpu/${PBS_JOBID}_${NNODES}_${NRANKS}_${RANKS_PER_NODE}_${BUF_SIZE}_py_hvd_profiling_out.txt
    date 
done 
