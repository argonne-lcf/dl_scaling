#!/bin/bash -x
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l daos=default
#PBS -A Aurora_deployment
#PBS -q alcf_daos_cn
#PBS -k doe 

#PBS -o  provide_path
#PBS -e  provide_path


# qsub -A Aurora_deployment -q EarlyAppAccess  ./job_lus.sh
# qsub -l select=1 -l walltime=00:30:00 -A Aurora_deployment -q alcf_daos_cn -l daos=default ./job_lus.sh  or - I 

echo $PBS_JOBID 
cd $PBS_O_WORKDIR
MY_PY_FILE=$MY_PY_FILE
echo $MY_PY_FILE

module restore
module unload intel_compute_runtime/release/agama-devel-551 
module unload intel_compute_runtime/release/stable-736.25
module unload oneapi/eng-compiler/2022.12.30.003

nnodes=$(cat $PBS_NODEFILE | wc -l)

date
source /gecko/Aurora_deployment/kaushik/soft/dso_testing/local-frameworks/source_env_lustre.sh
date

 
mkdir -p $PBS_O_WORKDIR/feb16-b/job_lus_strace_$nnodes

rpn=12
echo -e "Running python code"
date 
LOGDIR=$PBS_O_WORKDIR/feb16-b/job_lus_strace_$nnodes mpiexec -np $((rpn*nnodes)) -ppn $rpn /gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-strace-analyser/strace-analyzer/strace-wrapper.sh python3 $MY_PY_FILE
date


# rpn=12
# echo -e "Running python code"
# date 
# mpiexec -np $((rpn*nnodes)) -ppn $rpn python3 /gecko/CSC250STDM10_CNDA/kaushik/dso-workspace/init-test/ccl_profiling.py
# date

echo "Job Ended :  `date`"
exit 0
