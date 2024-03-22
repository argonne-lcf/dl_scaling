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
# qsub -l select=1 -l walltime=00:30:00 -A Aurora_deployment -q alcf_daos_cn -l daos=default ./job_slash_soft.sh  or - I 

echo $PBS_JOBID 
cd $PBS_O_WORKDIR
echo $MY_PY_FILE

module restore
date
module use /soft/modulefiles
module load frameworks
date
 

mkdir -p $PBS_O_WORKDIR/feb16-b/job_slash_strace_$nnodes

rpn=12
echo -e "Running python code"
date 
LOGDIR=$PBS_O_WORKDIR/feb16-b/job_slash_strace_$nnodes mpiexec -np $((rpn*nnodes)) -ppn $rpn /gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-strace-analyser/strace-analyzer/strace-wrapper-c.sh python3 $MY_PY_FILE
date


echo "Job Ended :  `date`"
exit 0
