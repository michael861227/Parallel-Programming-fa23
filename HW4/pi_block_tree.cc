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

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Count hit
    unsigned int seed = time(NULL) * world_rank;
    long long int local_tosses = tosses / world_size;
    int neighbor_rank;
    int layer = 0;
    int is_master;

    // TODO: binary tree redunction
    if (world_rank == 0) {
        local_tosses += tosses % world_size;
    }

    estimate_pi(local_tosses, seed);

    world_size >>= 1;

    while (world_size) {
        neighbor_rank = world_rank ^ (1 << layer);
        is_master = !(world_rank & (1 << layer));

        if (is_master) {
            // Recv
            long long int hit_holder;
            MPI_Recv(&hit_holder, 1, MPI_LONG_LONG_INT, neighbor_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            hit += hit_holder;
        }
        else {
            // Send
            MPI_Send(&hit, 1, MPI_LONG_LONG_INT, neighbor_rank, 0, MPI_COMM_WORLD);
            break;
        }

        ++layer;
        world_size >>= 1;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * (double)hit / tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
