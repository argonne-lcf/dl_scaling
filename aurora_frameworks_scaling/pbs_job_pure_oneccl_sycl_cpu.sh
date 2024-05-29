#!/bin/bash -x
# qsub -l nodes=2:ncpus=208 -q workq  -l walltime=02:00:00 -l filesystems=gila -A  Aurora_deployment ./pbs_job_
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

CPU_BINDING1=list:4:9:14:19:20:25:56:61:66:71:74:79
EXT_ENV="--env FI_CXI_DEFAULT_CQ_SIZE=1048576"
# https://oneapi-src.github.io/oneCCL/benchmark.html
APP1=/lus/gila/projects/CSC250STDM10_CNDA/kaushik/oneCCL/build/_install/examples/benchmark/benchmark

which python
 
mkdir -p ./out_${PBS_JOBID}/c_oneccl_cpu
for NNODES in 4 8 16 32 64 
do 
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))

    for BUF_SIZE in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216  33554432 67108864 134217728 268435456
    do
        date
        mpiexec ${EXT_ENV}  --env CCL_LOG_LEVEL=info  --env CCL_PROCESS_LAUNCHER=pmix  --env CCL_ATL_TRANSPORT=mpi \
                            --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind  $CPU_BINDING1    $APP1   \
                            --elem_counts ${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE},${BUF_SIZE}   \
                            --coll allreduce -j off -i 1 -w 0  --backend host  --sycl_dev_type host >  ./out_${PBS_JOBID}/c_oneccl_cpu/${PBS_JOBID}_${NNODES}_${NRANKS}_${RANKS_PER_NODE}_${BUF_SIZE}_sycl_ccl_cpu_out_w1.txt
        date
    echo ${BUF_SIZE}

    done
done

# For CPU only, --backend host --sycl_dev_type host