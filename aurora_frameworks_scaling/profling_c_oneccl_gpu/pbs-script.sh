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

module list
module use /soft/modulefiles
module load spack-pe-gcc  thapi
module load frameworks

cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"


export LD_LIBRARY_PATH=/gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-one-ccl/oneCCL/build/_install/lib:$LD_LIBRARY_PATH
source /gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-one-ccl/oneCCL/build/_install/env/setvars.sh

export LD_LIBRARY_PATH=/soft/compilers/oneapi/2023.12.15.001/oneapi/ccl/2021.11/lib:$LD_LIBRARY_PATH
source /soft/compilers/oneapi/2023.12.15.001/oneapi/ccl/2021.11/vars.sh
source /soft/compilers/oneapi/2023.12.15.001/oneapi/ccl/latest/env/vars.sh


# mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose ./cpu_allreduce_test
iprof  --max-name-size -1 mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} gpu_tile_compact.sh  ./cpu_allreduce_test  
mpiexec --np 4 -ppn 4 ./cpu_allreduce_test
