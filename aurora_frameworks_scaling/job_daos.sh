#!/bin/bash -x
#PBS -l select=1
#PBS -l walltime=03:30:00
#PBS -l daos=default
#PBS -A Aurora_deployment
#PBS -q alcf_daos_cn
#PBS -k doe 

# qsub -A Aurora_deployment -q alcf_daos_cn  ./job_daos.sh
# qsub -l select=1 -l walltime=01:30:00 -A Aurora_deployment -q alcf_daos_cn -l daos=default ./job_daos.sh  or - I 



# After creating the container, keep all the frameworks related files in a similar location as below and 
# module unload the main packages and module load from the below location. 
# /tmp/CSC250STDM10_CNDA/datascience_softwares/to-go-daos/local-frameworks/

# use the tool mentioned in https://docs.daos.io/v2.4/user/datamover/
# for fast copy from lustre to daos 




set -x
echo "Job Started :  `date`"
echo $PBS_JOBID
cd $PBS_O_WORKDIR
echo "List of allocated nodes : $PBS_NODEFILE"

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





# export D_LOG_MASK=INFO  
# export D_LOG_STDERR_IN_LOG=1
# export D_LOG_FILE="$PBS_O_WORKDIR/ior-p.log" 
# export D_IL_REPORT=1 # Logs for IL
# LD_PRELOAD=$DAOS_PRELOAD mpiexec 


echo -e "sourcing module from contianer"
module list
start_time=$(date +%s)
source /tmp/${DAOS_POOL}/${DAOS_CONT}/local-frameworks/source_env_daos.sh
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "source local frameworks time $diff seconds."
module list

start_time=$(date +%s)
module use /soft/modulefiles/daos/
module load base
end_time=$(date +%s)
diff=$(($end_time - $start_time))
echo "2nd daos module load time $diff seconds."
module avail
module list


echo -e "Running python code"
LD_PRELOAD=/usr/lib64/libpil4dfs.so 
rpn=12   
nnodes=$(cat $PBS_NODEFILE | wc -l)
# start_time=$(date +%s)
# mpiexec -np $((rpn*nnodes)) -ppn $rpn --cpu-bind "list:0:1:2:3:52:53:54:55:56:57:58:59" -genvall --no-vni --env LD_PRELOAD=/usr/lib64/libpil4dfs.so  python3 /gecko/CSC250STDM10_CNDA/kaushik/dso-workspace/init-test/one_line_torch.py
# end_time=$(date +%s)
# diff=$(($end_time - $start_time))
# echo "Total python run time $diff seconds."

mkdir -p $PBS_O_WORKDIR/job_lus_strace_$nnodes
date 
LOGDIR=$PBS_O_WORKDIR/job_lus_strace_$nnodes mpiexec -np $((rpn*nnodes)) -ppn $rpn --cpu-bind "list:0:1:2:3:52:53:54:55:56:57:58:59"  -genvall --no-vni --env LD_PRELOAD=/usr/lib64/libpil4dfs.so /gecko/CSC250STDM10_CNDA/kaushik/gitrepos/src-strace-analyser/strace-analyzer/strace-wrapper-c.sh python3 /gecko/CSC250STDM10_CNDA/kaushik/dso-workspace/init-test/one_line_torch.py
date

clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT} # cleanup dfuse mounts
# daos container destroy  ${DAOS_POOL} ${DAOS_CONT} 
date
echo "Job Ended :  `date`"
qstat -xf $PBS_JOBID
echo "done"
exit 0