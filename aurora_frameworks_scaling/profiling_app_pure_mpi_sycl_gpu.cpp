// module use /soft/modulefiles
// module load frameworks
// mpic++ -o sycl_cpu sycl_cpu.cpp -fsycl -lmpi -I/opt/aurora/24.086.0/oneapi/compiler/2024.1/include/sycl/
// unset ONEAPI_DEVICE_SELECTOR

#include <sycl/sycl.hpp> 
#include <mpi.h>
#include <cmath>
#include <chrono>
using namespace sycl;

int main(int argc, char** argv) 
{
    int rank;
    double t1, t2, t3, t4, init_timer;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    MPI_Barrier( MPI_COMM_WORLD ); 
 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sycl::queue Q(sycl::gpu_selector_v);
    
    // std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>()  << std::endl;
    int  global_range;
    if (argc == 2)  
    {
       global_range = atoi(argv[1])/4;
    }
    else
    {
        global_range = 1048576;
    }
    auto *A  = sycl::malloc_device<float>(global_range,Q);
    auto *B  = sycl::malloc_device<float>(global_range,Q);

    Q.parallel_for(global_range, [=](sycl::item<1> id) { A[id] = 1; }).wait();
    Q.parallel_for(global_range, [=](sycl::item<1> id) { B[id] = 0; }).wait();    
    MPI_Barrier( MPI_COMM_WORLD ); 


    std::vector<double> elapsed(50);
    for (int i = 0; i < 50; i++)
    {
        t3 = MPI_Wtime();
        MPI_Allreduce(A, B, global_range, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD ); 
        t4 = MPI_Wtime();
        if ( rank == 0 )    elapsed[i]=( t4 - t3 ) * 1e6;
    }

    // Q.memcpy(B_host.data(),B, global_range*sizeof(float)).wait();
    // for (size_t i = 0; i < global_range; i++)
    //     std::cout << "B[ " << i << " ] = " << B_host[i] << std::endl;

    if ( rank == 0 )    
    {
        for (int i = 0; i < 50; i++)
        {
            std::cout<<elapsed[i]<<std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
