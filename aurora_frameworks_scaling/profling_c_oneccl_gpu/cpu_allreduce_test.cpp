#include <iostream>
#include <mpi.h>
#include "oneapi/ccl.hpp"

// set CCL_ZE_IPC_EXCHANGE=sockets is not necessary according to our intel contact. 
//Try something like this CCL_PROCESS_LAUNCHER=pmix  mpiexec -np 4 -ppn 4 --pmi=pmix ./cpu_allreduce_test We are using PALS job launcher with the PMIx API in Aurora


using namespace std;
int main() 
{
    ccl::init();
    return 0;
}
