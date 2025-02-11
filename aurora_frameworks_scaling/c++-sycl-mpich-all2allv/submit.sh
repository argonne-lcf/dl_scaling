#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N mpi_all2allv
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -A Aurora_deployment
#PBS -k doe
#PBS -j oe

# Change to working directory
cd ${PBS_O_WORKDIR}

module load frameworks

EXE=/flare/Aurora_deployment/balin/oneCCL/all2allv/mpi/all2allv_mpi_mod
AFFINITY=/flare/Aurora_deployment/balin/oneCCL/affinity.sh
NNODES=`wc -l < $PBS_NODEFILE`
#RANKS_PER_NODE=2
#NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING=list:1:8:16:24:32:40:53:60:68:76:84:92
EXT_ENV="--env FI_CXI_DEFAULT_CQ_SIZE=1048576"
OTHER_BUF_SIZE=0

for RANKS_PER_NODE in 2 4 6 12
do
  NRANKS=$(( NNODES * RANKS_PER_NODE ))
  if [ ! -d $RANKS_PER_NODE ]; then
    mkdir $RANKS_PER_NODE
  fi
  cd $RANKS_PER_NODE

  for BUF_SIZE in 1024 2048 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216  33554432 67108864 134217728 268435456
  do
    date
    echo Running with ${BUF_SIZE} and $RANKS_PER_NODE
    mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING  $AFFINITY \
        $EXE ${BUF_SIZE} ${OTHER_BUF_SIZE} 2>&1 | tee ./${PBS_JOBID}_N${NNODES}_n${NRANKS}_ppn${RANKS_PER_NODE}_b${BUF_SIZE}_mpi.txt
    date
  done

  cd ..
done


