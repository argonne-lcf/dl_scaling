// mpicc -o profiling_app_pure_mpi_cpu profiling_app_pure_mpi_cpu.c -lmpi 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char * argv[])
{
    int rank, nranks;
    double t1, t2, t3, t4;
    long int li;
    long long int p1, p2, values[2], c1, c2, i1, i2;
    int N_AllRed[2] = { 8, 2048 }, N_AllMax = 2048, N_AllBst[2] = { 16, 8192 }, N_BstMax = 8192;
    
    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nranks );

    float s_reduce[16384][16384];
    float r_reduce[16384][16384];

    for ( int i = 0; i < 16384; i++ )
    {
        for ( int j = 0; j < 16384; j++ )
        {
            s_reduce[i][j]=1;
            r_reduce[i][j]=0;
        }
    }
    
    // for ( int i = 0; i < 1; i++ )
    // {
    //     for ( int j = 0; j < 8; j++ )
    //     {
    //         printf( "Before all_reduce Rank %2d value of r_reduce   : %i \n", rank,  r_reduce[i][j] );
    //         printf( "Before all_reduce Rank %2d value of s_reduce   : %i \n", rank,  s_reduce[i][j] );
    //     }
    // }
    
    MPI_Barrier( MPI_COMM_WORLD ); 
    t1 = MPI_Wtime();

    MPI_Allreduce( s_reduce, r_reduce, 134217728, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );

    MPI_Barrier( MPI_COMM_WORLD ); 
    t2 = MPI_Wtime();

       if ( rank == 0 )   printf("First all reduce time : %8.4lf u.sec \n", ( t2 - t1 ) * 1e6 );


    MPI_Barrier( MPI_COMM_WORLD ); 
    t3 = MPI_Wtime();

    MPI_Allreduce( s_reduce, r_reduce, 134217728, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );

    MPI_Barrier( MPI_COMM_WORLD ); 
    t4 = MPI_Wtime();
    
    if ( rank == 0 )  printf("Second all reduce time : %8.4lf u.sec  \n", ( t4 - t3 ) * 1e6 );


    // for ( int i = 0; i < 1; i++ )
    // {
    //     for ( int j = 0; j < 8; j++ )
    //     {
    //         printf( "After all_reduce Rank %2d value of r_reduce   : %i \n", rank,  r_reduce[i][j] );
    //         printf( "After all_reduce Rank %2d value of s_reduce   : %i \n", rank,  s_reduce[i][j] );
    //     }
    // }


    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();

    return 0;
}
