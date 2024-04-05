#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N_BARRIER 3000    /* 15us per barrier, 45 seconds measurement */
#define N_REDUCE  1000    /* 70us per allresuce, 70 seconds measurement */
#define N_BCAST   1800    /* 50us per bcast, 70 seconds measurement */

int main(int argc, char * argv[])
{
    int rank, nranks, rank_half_comm, nranks_half_comm;
    char *s_reduce, *r_reduce, *b_bcast;
    double t1, t2, t3;
    long int li;
    MPI_Comm HALF_comm;
    int N_AllRed[2] = { 8, 2048 }, N_AllMax = 2048, N_AllBst[2] = { 16, 8192 }, N_BstMax = 8192;
    
    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nranks );

    if ( nranks < 2 ) 
    {   
        fprintf( stderr, "This benchmark requires more than 1 MPI process, exit...\n" );
        MPI_Finalize();
        return 1;
    }
    

    /* Split COMM_WORLD in half: 0..nranks/2-1, nranks/2..nranks-1 */
    MPI_Comm_split( MPI_COMM_WORLD, ( rank < ( nranks / 2 ) ), 0, &HALF_comm );
    MPI_Comm_rank ( HALF_comm, &rank_half_comm );
    MPI_Comm_size ( HALF_comm, &nranks_half_comm );

    
    /************************************ Barrier *************************************/
    MPI_Barrier( MPI_COMM_WORLD ); 
    t1 = MPI_Wtime();

    for ( int i = 0; i < N_BARRIER; i++ )
        MPI_Barrier( MPI_COMM_WORLD ); 

    MPI_Barrier( MPI_COMM_WORLD ); 
    t2 = MPI_Wtime();
    
    for ( int i = 0; i < N_BARRIER; i++ )
        MPI_Barrier( HALF_comm ); 

    MPI_Barrier( HALF_comm ); 
    t3 = MPI_Wtime();

    if ( rank == 0 )           printf( "%4d: Barrier COMM_WORLD, us: %8.4lf\n", nranks, ( t2 - t1 ) * 1e6 / (double)N_BARRIER );
    if ( rank_half_comm == 0 ) printf( "%4d: Barrier HALF, us      : %8.4lf\n", nranks_half_comm, ( t3 - t2 ) * 1e6 / (double)N_BARRIER );

    /************************************ Allreduce 8 and 2048 bytes ***********************************/
    s_reduce = (char *)malloc( N_AllMax * sizeof( char ) );
    r_reduce = (char *)malloc( N_AllMax * sizeof( char ) );
    if ( ( s_reduce == NULL ) || ( r_reduce == NULL ) )
    {
        fprintf( stderr, "Failed to allocate buffers for Allreduce test, exit...\n" );
        MPI_Finalize();
        return 2;
    }

    /* should we use memset for it? */
    for ( int i = 0; i < N_AllMax; i++ ) 
    {
        s_reduce[i] = (char)rank;
        r_reduce[i] = 0;
    }
    
    MPI_Barrier( MPI_COMM_WORLD ); 
    t1 = MPI_Wtime();
    MPI_Allreduce( r_reduce, s_reduce, 8, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD ); 
    t2 = MPI_Wtime();
    if ( rank == 0 ) printf( "First call: Allreduce %4d B COMM_WORLD, us: %8.4lf\n", 8, ( t2 - t1 ) * 1e6 );
    
    for ( int j = 0; j < 2; j++ )
    {

        MPI_Barrier( MPI_COMM_WORLD ); 
        t1 = MPI_Wtime();
    
        for ( int i = 0; i < N_REDUCE; i++ )
            MPI_Allreduce( r_reduce, s_reduce, N_AllRed[j], MPI_CHAR, MPI_SUM, MPI_COMM_WORLD );
    
        MPI_Barrier( MPI_COMM_WORLD ); 
        t2 = MPI_Wtime();
    
        /* Correctness: each r_reduce[i] value must be the same for any i, sum of them = r_reduce[0] * 2048 */
        li = 0;
        for ( int i = 0; i < N_AllRed[j]; i++ )
            li = li + (long int)r_reduce[i];
        if ( li !=  ( (long int)r_reduce[0] * N_AllRed[j] ) ) fprintf( stderr, "Rank %6d failed Allreduce correctness tests\n", rank );
    
        if ( rank == 0 )           printf( "%4d: Allreduce %4d B COMM_WORLD, us: %8.4lf\n", nranks, N_AllRed[j], ( t2 - t1 ) * 1e6 / (double)N_REDUCE );
    
        MPI_Barrier( HALF_comm ); 
        t2 = MPI_Wtime();
        
        for ( int i = 0; i < N_REDUCE; i++ )
            MPI_Allreduce( r_reduce, s_reduce, N_AllRed[j], MPI_CHAR, MPI_SUM, HALF_comm );
        
        MPI_Barrier( HALF_comm ); 
        t3 = MPI_Wtime();
        
        li = 0;
        for ( int i = 0; i < N_AllRed[j]; i++ )
            li = li + (long int)r_reduce[i];
        if ( li !=  ( (long int)r_reduce[0] * N_AllRed[j] ) ) fprintf( stderr, "Rank %6d failed Allreduce correctness tests\n", rank );
    
        if ( rank_half_comm == 0 ) printf( "%4d: Allreduce %4d B HALF, us      : %8.4lf\n", nranks_half_comm, N_AllRed[j], ( t3 - t2 ) * 1e6 / (double)N_REDUCE );
    }
    

    free( s_reduce );
    free( r_reduce );


    MPI_Finalize();

    return 0;
}
