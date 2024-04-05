#!/bin/bash
 
# mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose ./start_gprofng.sh 1
# mpiexec --np 4 -ppn 4 ./test0


export PATH=/home/morozov/binutils-2.39/bin:$PATH
export LD_LIBRARY_PATH=/home/morozov/papi-20221031/lib:$LD_LIBRARY_PATH
/home/morozov/binutils-2.39/bin/gprofng collect app -O experiment.$1.$PALS_RANKID.er ./test0

