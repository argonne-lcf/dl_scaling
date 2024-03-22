qsub -l select=1    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh
qsub -l select=2    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh 
qsub -l select=4    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh
qsub -l select=8    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh 
qsub -l select=16   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh
qsub -l select=32   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh 
qsub -l select=64   -l walltime=02:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh
qsub -l select=128   -l walltime=03:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh 
qsub -l select=256   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh
qsub -l select=512   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_ccl.sh 

qsub -l select=1    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh
qsub -l select=2    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh 
qsub -l select=4    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh
qsub -l select=8    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh 
qsub -l select=16   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh
qsub -l select=32   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh 
qsub -l select=64   -l walltime=00:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh
qsub -l select=128   -l walltime=00:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh 
qsub -l select=256   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh
qsub -l select=512   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_hvd.sh 
 
qsub -l select=1    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh
qsub -l select=2    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh  
qsub -l select=4    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh
qsub -l select=8    -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh  
qsub -l select=16   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh
qsub -l select=32   -l walltime=01:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh  
qsub -l select=64   -l walltime=02:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh
qsub -l select=128   -l walltime=03:30:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh  
qsub -l select=256   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh
qsub -l select=512   -l walltime=01:00:00  -k doe -A Aurora_deployment -q EarlyAppAccess ./job_tmp_soft_mpi.sh  

