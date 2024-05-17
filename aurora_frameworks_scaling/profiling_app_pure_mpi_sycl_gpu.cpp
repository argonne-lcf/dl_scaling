// module use /soft/modulefiles
// module load frameworks/2023.12.15.001 
// mpic++ -o profiling_app_pure_mpi_sycl_gpu profiling_app_pure_mpi_sycl_gpu.cpp -fsycl -lmpi -I/opt/aurora/23.275.2/oneapi/compiler/2024.0/include/sycl

#include <sycl/sycl.hpp> 
#include <mpi.h>
#include <cmath>

using namespace sycl;

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    int rank;
    double t1, t2, t3, t4;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sycl::queue Q(sycl::gpu_selector_v);
    // sycl::queue Q(sycl::cpu_selector_v);
    
    std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>()  << std::endl;
    
    int  global_range;
    if (argc == 2)  
    {
       global_range = atoi(argv[1]);
    }
    else
    {
        global_range = 1048576;
    }
    
    
    
    auto *A  = sycl::malloc_device<float>(global_range,Q);
    auto *B  = sycl::malloc_device<float>(global_range,Q);

    std::vector<float> A_host(global_range);
    std::vector<float> B_host(global_range);

    Q.parallel_for(global_range, [=](sycl::item<1> id) { A[id] = 1; }).wait();
    Q.parallel_for(global_range, [=](sycl::item<1> id) { B[id] = 0; }).wait();    

    MPI_Barrier( MPI_COMM_WORLD ); 
    t1 = MPI_Wtime();
    MPI_Allreduce(A, B, global_range, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier( MPI_COMM_WORLD ); 
    t2 = MPI_Wtime();
    if ( rank == 0 )   printf("First all reduce time : %8.4lf u.sec for bytes %d by rank %d \n", ( t2 - t1 ) * 1e6 , global_range, rank );

    for (int i = 0; i < 10; i++)
    {
        MPI_Barrier( MPI_COMM_WORLD ); 
        t1 = MPI_Wtime();
        MPI_Allreduce(A, B, global_range, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD ); 
        t2 = MPI_Wtime();
        if ( rank == 0 )   printf("Second all reduce time : %8.4lf u.sec for bytes %d by rank %d \n", ( t2 - t1 ) * 1e6 , global_range, rank);
    }

    // Q.memcpy(B_host.data(),B, global_range*sizeof(float)).wait();
    // for (size_t i = 0; i < global_range; i++)
    //     std::cout << "B[ " << i << " ] = " << B_host[i] << std::endl;
    
    MPI_Finalize();
    return 0;
}
