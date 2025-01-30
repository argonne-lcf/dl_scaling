#include <sycl/sycl.hpp> 
#include <mpi.h>
#include <cmath>
#include <chrono>

int main(int argc, char** argv) 
{
    int rank, size;
    double t1, t2, t3, t4, init_timer;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    MPI_Barrier( MPI_COMM_WORLD ); 
 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sycl::queue Q(sycl::gpu_selector_v);
    
    std::cout << "Rank " << rank << " running on " << Q.get_device().get_info<sycl::info::device::name>()  << std::endl;
    int  global_range, elements_per_proc;
    if (argc == 2)  
    {
       elements_per_proc = atoi(argv[1])/4;
    }
    else
    {
        elements_per_proc = 1048576;
    }
    global_range = elements_per_proc * size;

    auto *send_buff  = sycl::malloc_device<float>(global_range,Q);
    auto *rcv_buff  = sycl::malloc_device<float>(global_range,Q);
    Q.parallel_for(global_range, [=](sycl::item<1> id) { send_buff[id] = rank; }).wait();
    Q.parallel_for(global_range, [=](sycl::item<1> id) { rcv_buff[id] = 0; }).wait();    
    MPI_Barrier( MPI_COMM_WORLD ); 

    int iters = 50;
    std::vector<double> elapsed(iters);
    for (int i = 0; i < iters; i++)
    {
        t3 = MPI_Wtime();
        MPI_Alltoall(send_buff, elements_per_proc, MPI_FLOAT, 
                     rcv_buff, elements_per_proc, MPI_FLOAT, 
                     MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD ); 
        t4 = MPI_Wtime();
        if ( rank == 0 )    elapsed[i]=( t4 - t3 ) * 1e3;
    }

    // Q.memcpy(B_host.data(),B, global_range*sizeof(float)).wait();
    // for (size_t i = 0; i < global_range; i++)
    //     std::cout << "B[ " << i << " ] = " << B_host[i] << std::endl;

    double avg = 0.0;
    int skip = 4;
    if ( rank == 0 )    
    {
        for (int i = skip; i < iters; i++)
        {
            avg = avg + elapsed[i];
            std::cout<<elapsed[i]<<std::endl;
        }
        avg = avg / (iters - skip);
        std::cout << "Average all2all time: " << avg << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
