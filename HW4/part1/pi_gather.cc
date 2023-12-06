#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

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

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Count hit
    unsigned int seed = time(NULL) * world_rank;
    long long int local_tosses = tosses / world_size;
    long long int hits[world_size];

    if (world_rank == 0)
    {
        local_tosses += tosses % world_size;
    }

    estimate_pi(local_tosses, seed);

    // TODO: use MPI_Gather
    MPI_Gather(&hit, 1, MPI_LONG_LONG_INT, hits, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for (int i = 1; i < world_size; i++)
        {
            hit += hits[i];
        }

        pi_result = 4.0 * hit / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
