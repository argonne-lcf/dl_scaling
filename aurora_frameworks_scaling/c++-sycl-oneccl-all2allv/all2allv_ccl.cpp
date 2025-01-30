#include <sycl/sycl.hpp> 
#include <mpi.h>
#include <cmath>
#include <chrono>
#include "oneapi/ccl.hpp"

// Get the nearest neighbors
std::vector<int> get_nearest_neighbors(int rank, int size) 
{
    std::vector<int> neighbors;
    int num_neighbors = 1;

    if (size == 2) {
        int other_rank = (rank - 1 + size) % size;
        neighbors.push_back(other_rank);
    } else {
        for (int i = 0; i < num_neighbors; i++) {
            int left_rank = (rank - (1 + i) + size) % size;
            int right_rank = (rank + (1 + i)) % size;
            neighbors.push_back(left_rank);
            neighbors.push_back(right_rank);
        }
    }

    std::cout << "Rank " << rank << " neighbor list: ";
    for (int n : neighbors) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    return neighbors;
}

int main(int argc, char** argv) 
{
    int rank, size;
    double t1, t2, t3, t4, init_timer;

    ccl::init();

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    MPI_Init(&argc, &argv);
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    MPI_Barrier( MPI_COMM_WORLD ); 
 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    sycl::queue Q(sycl::gpu_selector_v);
    
    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(Q.get_device());
    auto ctx = ccl::create_context(Q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(Q);

    std::cout << "Rank " << rank << " running on " << Q.get_device().get_info<sycl::info::device::name>()  << std::endl;
    int  elements_per_proc;
    if (argc == 2)  
    {
       elements_per_proc = atoi(argv[1])/4;
    }
    else
    {
        elements_per_proc = 1048576;
    }

    // Get the neighboring ranks
    std::vector<int> neighbors;
    neighbors = get_nearest_neighbors(rank, size);
    MPI_Barrier( MPI_COMM_WORLD );

    // Initialize arrays
    std::vector<float> send_buff;
    std::vector<unsigned long> send_counts(size,0);
    std::vector<unsigned long> send_displs(size,0);
    std::vector<unsigned long> rcv_counts(size,0);
    std::vector<unsigned long> rcv_displs(size,0);

    // Fill in the send counts, displacements and buffers
    int global_send_elements = 0;
    for (int neighbor : neighbors)
    {
        send_counts[neighbor] = elements_per_proc;
        //global_send_elements += elements_per_proc;
        send_displs[neighbor] = global_send_elements;
        global_send_elements += elements_per_proc;
        for (int n = 0; n < elements_per_proc; n++) {
            send_buff.push_back(rank);
        }
    }
    std::cout << "Rank " << rank << " sending " << global_send_elements << " elements" << std::endl;
    fflush(stdout);
    MPI_Barrier( MPI_COMM_WORLD );

    // Get the received data
    int global_rcv_elements = 0;
    MPI_Alltoall(send_counts.data(), 1, MPI_UNSIGNED_LONG, 
                 rcv_counts.data(), 1, MPI_UNSIGNED_LONG, 
                 MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        if (rcv_counts[i] != 0) {
                std::cout << "Rank " << rank << " receives " << rcv_counts[i] <<
                          " elements from rank " << i << std::endl;
        }
        //global_rcv_elements += rcv_counts[i];
        rcv_displs[i] = global_rcv_elements;
        global_rcv_elements += rcv_counts[i];
    }
    std::vector<float> rcv_buff(global_rcv_elements, -1.0);

    // Move the send and receive buffers to the GPU
    float *dsend_buff  = sycl::malloc_device<float>(global_send_elements,Q);
    float *drcv_buff  = sycl::malloc_device<float>(global_rcv_elements,Q);
    Q.memcpy((void *) dsend_buff, (void *) send_buff.data(), global_send_elements*sizeof(float));
    Q.memcpy((void *) drcv_buff, (void *) rcv_buff.data(), global_rcv_elements*sizeof(float));  
    Q.wait();
    MPI_Barrier( MPI_COMM_WORLD ); 

    int iters = 50;
    std::vector<double> elapsed(iters);
    for (int i = 0; i < iters; i++)
    {
        t3 = MPI_Wtime();
        //MPI_Alltoallv(dsend_buff, send_counts.data(), send_displs.data(), MPI_FLOAT, 
        //             drcv_buff, rcv_counts.data(), rcv_displs.data(), MPI_FLOAT, 
        //             MPI_COMM_WORLD);
        ccl::alltoallv(dsend_buff, send_counts, 
                       drcv_buff, rcv_counts, 
                       comm, stream).wait();
        MPI_Barrier( MPI_COMM_WORLD ); 
        t4 = MPI_Wtime();
        if ( rank == 0 )    elapsed[i]=( t4 - t3 ) * 1e3;
    }

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

    /*
    Q.memcpy(rcv_buff.data(), drcv_buff, global_rcv_elements*sizeof(float)).wait();    
    if (rank == 0) {
        std::cout << "Rank 0 received: " << std::endl;
        for (float n : rcv_buff) {
            std::cout << n << std::endl;
        }
    }
    */

    MPI_Finalize();
    return 0;
}
