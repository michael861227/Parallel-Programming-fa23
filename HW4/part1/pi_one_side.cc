#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")

long long int hit;

void estimate_pi(long long int tosses, unsigned int seed)
{
    double x, y, distance_squared;
    for (long long int i = 0; i < tosses; i++)
    {
        x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            hit++;
    }
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Count hit
    long long int local_tosses = tosses / world_size;
    unsigned int seed = time(NULL) * world_rank;
    long long int *gather;

    if (world_rank == 0) {
        local_tosses += tosses % world_size;
    }

    estimate_pi(local_tosses, seed);

    // MPI one-side
    if (world_rank == 0)
    {
        // Master
        MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &gather);
        *gather = hit;
        MPI_Win_create(gather, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&hit, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (*gather) / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
        // Free Memory
        MPI_Free_mem(gather);
    }
    
    MPI_Finalize();
    return 0;
}