#!/bin/bash -x
#PBS -l select=1
#PBS -l walltime=03:30:00
#PBS -l daos=default
#PBS -A Aurora_deployment
#PBS -q alcf_daos_cn
#PBS -k doe 

# qsub -l select=1 -l walltime=01:30:00 -A Aurora_deployment -q alcf_daos_cn -l daos=default ./job_daos.sh  or - I 

# After creating the container, keep all the software frameworks related files inside the container. 
# module unload the main packages from /soft and module load from the container /tmp/CSC250STDM10_CNDA/datascience_softwares/to-go-daos/local-frameworks/
# use the tool mentioned in https://docs.daos.io/v2.4/user/datamover/ for fast copy from lustre to daos 


set -x
echo "Job Started :  `date`"
echo $PBS_JOBID
cd $PBS_O_WORKDIR
echo "List of allocated nodes : $PBS_NODEFILE"

# Load the DAOS modules and mount the container

module restore
start_time=$(date +%s)
module use /soft/modulefiles
module load daos/base
module load mpich/51.2/icc-all-pmix-gpu 
module list
env|grep DRPC
export DAOS_POOL=CSC250STDM10_CNDA
export DAOS_CONT=datascience-softwares
# daos container create --type POSIX --dir-oclass=S1 --file-oclass=SX ${DAOS_POOL} ${DAOS_CONT}
daos container get-prop ${DAOS_POOL} ${DAOS_CONT}
daos cont      query  ${DAOS_POOL} ${DAOS_CONT}
clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT} 
launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "1st module load + launch time $diff seconds."
mount|grep dfuse

# To enable DAOS logs

# export D_LOG_MASK=INFO  
# export D_LOG_STDERR_IN_LOG=1
# export D_LOG_FILE="$PBS_O_WORKDIR/ior-p.log" 
# export D_IL_REPORT=1 # Logs for IL
# LD_PRELOAD=$DAOS_PRELOAD mpiexec 

# module load the frameworks software from the container instead of the main softwares. 

echo -e "sourcing module from contianer"
module list
start_time=$(date +%s)
source /tmp/${DAOS_POOL}/${DAOS_CONT}/local-frameworks/source_env_daos.sh
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "source local frameworks time $diff seconds."
module list

# Temporary patch to make sure the module daos is unaffected by the previous module load

start_time=$(date +%s)
module use /soft/modulefiles/daos/
module load base
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "2nd daos module load time $diff seconds."
module avail
module list

# Run the app from within the DAOS container

echo -e "Running python code"
LD_PRELOAD=/usr/lib64/libpil4dfs.so 
CPU_BINDING1=list:4:9:14:19:20:25:56:61:66:71:74:79
EXT_ENV="--env FI_CXI_DEFAULT_CQ_SIZE=1048576"

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

BUF_SIZE=8
start_time=$(date +%s)
mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1 -genvall --no-vni --env LD_PRELOAD=/usr/lib64/libpil4dfs.so  python3 pbs_job_py_torch_ccl_gpu.py $BUF_SIZE
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "Total python run time $diff seconds."

# For strace

# mkdir -p $PBS_O_WORKDIR/job_lus_strace_${NNODES}
# date 
# LOGDIR=$PBS_O_WORKDIR/job_lus_strace_${NNODES} mpiexec ${EXT_ENV} --np ${NRANKS} -ppn ${RANKS_PER_NODE}  --cpu-bind  $CPU_BINDING1 -genvall --no-vni --env LD_PRELOAD=/usr/lib64/libpil4dfs.so gitrepos/src-strace-analyser/strace-analyzer/strace-wrapper-c.sh  python3 pbs_job_py_torch_ccl_gpu.py $BUF_SIZE
# date

# To unmount your container

clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT} # cleanup dfuse mounts
# daos container destroy  ${DAOS_POOL} ${DAOS_CONT} 
date
echo "Job Ended :  `date`"
qstat -xf $PBS_JOBID
echo "done"
exit 0
