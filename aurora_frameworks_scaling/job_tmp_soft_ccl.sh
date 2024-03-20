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
# qsub -l select=1 -l walltime=00:30:00 -A Aurora_deployment -q alcf_daos_cn -l daos=default ./job_tmp_soft_ccl.sh  or - I 
#  qsub -l select=1   -l walltime=00:30:00 -o 1node-lus          -k doe -A Aurora_deployment -q EarlyAppAccess -l daos=default ./job_lus.sh

echo $PBS_JOBID 
cd $PBS_O_WORKDIR

module restore
module unload intel_compute_runtime/release/agama-devel-551 
module unload intel_compute_runtime/release/stable-736.25
module unload oneapi/eng-compiler/2022.12.30.003

nnodes=$(cat $PBS_NODEFILE | wc -l)
rpn=1

echo -e "Transferring and sourcing the modules from /tmp"
date
mpiexec -np $((rpn*nnodes)) -ppn $rpn --pmi=pmix tar zxf ~/soft/local-frameworks-23.266.2-20240131a.tar.gz 	-C /tmp 
echo "end untar to tmp: 'date'"
date
source /tmp/local-frameworks/source_env.sh
date

rpn=12
echo -e "Running python code"
date 
mpiexec --env FI_CXI_DEFAULT_CQ_SIZE=16384  --env FI_CXI_OVFLOW_BUF_SIZE=8388608 --env FI_CXI_CQ_FILL_PERCENT=20 -np $((rpn*nnodes)) -ppn $rpn python3 ccl_profiling.py 
date 

echo "Job Ended :  `date`"
exit 0
