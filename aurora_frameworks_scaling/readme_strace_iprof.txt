For strace 

https://github.com/cniethammer/strace-analyzer
mkdir -p $PBS_O_WORKDIR/1job_slash_strace_$nnodes
date 
LOGDIR=$PBS_O_WORKDIR/job_lus_strace_${NNODES} mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1 gitrepos/src-strace-analyser/strace-analyzer/strace-wrapper-c.sh  python3 pbs_job_py_torch_ccl_gpu.py $BUF_SIZE
date


For Thapi 

module use /soft/modulefiles
module load spack-pe-gcc  thapi
iprof  --max-name-size -1 mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1 python3 pbs_job_py_torch_ccl_gpu.py $BUF_SIZE
