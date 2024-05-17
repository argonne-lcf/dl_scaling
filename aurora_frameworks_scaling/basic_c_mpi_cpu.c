// /opt/aurora/23.275.2/CNDA/mpich/52.2/mpich-ofi-all-icc-default-pmix-gpu-drop52/lib/libmpi.so.12

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char * argv[])
{
    int rank, nranks;
    double t1, t2, t3, t4;
    long int li;
    
    int  global_range;
    if (argc == 2)  
    {
       global_range = atoi(argv[1])/4;
    }
    else
    {
        global_range = 1048576;
    }
    
    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nranks );

    float s_reduce[268435456];
    float r_reduce[268435456];

    for ( int i = 0; i < 268435456; i++ )
    {
            s_reduce[i]=1;
            r_reduce[i]=0;
    }
    

    double elapsed[100];

    MPI_Barrier( MPI_COMM_WORLD ); 

    for (int i = 0; i < 100; i++)
    {
        t3 = MPI_Wtime();
        MPI_Allreduce( s_reduce, r_reduce, global_range, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
        t4 = MPI_Wtime();
        if ( rank == 0 )   
            elapsed[i] = ( t4 - t3 ) * 1e6 ;
    }


    MPI_Barrier( MPI_COMM_WORLD );

    for (int i = 0; i < 100; i++)
    {
        if ( rank == 0 )    printf("%lf \n", elapsed[i]);
    }

    MPI_Finalize();
    return 0;
}
