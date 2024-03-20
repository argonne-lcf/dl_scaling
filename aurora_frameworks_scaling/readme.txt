

For strace 

https://github.com/cniethammer/strace-analyzer

nnodes=$(cat $PBS_NODEFILE | wc -l)
mkdir -p $PBS_O_WORKDIR/1job_slash_strace_$nnodes
rpn=12
echo -e "Running python code"
date 
LOGDIR=$PBS_O_WORKDIR/1job_slash_strace_$nnodes mpiexec -np $((rpn*nnodes)) -ppn $rpn /gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-profilers/src-strace-analyser/strace-analyzer/strace-wrapper-c.sh python3 /gecko/CSC250STDM10_CNDA/kaushik/dso-workspace/strace_out/feb29/ccl_profiling_1.py
date
echo "Job Ended :  `date`"
exit 0



For Thapi 

module use /soft/modulefiles
module load spack-pe-gcc  thapi

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
#export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=software
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"
export MPIR_CVAR_CH4_MT_MODEL=lockless
export MPIR_CVAR_ENABLE_GPU=1 
echo MPIR_CVAR_ENABLE_GPU=1


# mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose ./collectives
iprof  --max-name-size -1 mpiexec --env FI_CXI_DEFAULT_CQ_SIZE=16384  --env FI_CXI_OVFLOW_BUF_SIZE=8388608 --env FI_CXI_CQ_FILL_PERCENT=20 --np ${NRANKS} -ppn ${RANKS_PER_NODE} gpu_tile_compact.sh  ./collectives  

# #qsub -l select=1 -l place=scatter -l walltime=00:60:00 -A Aurora_deployment   -q EarlyAppAccess -k doe 




For large number of nodes 

mpiexec --env FI_CXI_DEFAULT_CQ_SIZE=16384  --env FI_CXI_OVFLOW_BUF_SIZE=8388608 --env FI_CXI_CQ_FILL_PERCENT=20 -np $((rpn*nnodes))