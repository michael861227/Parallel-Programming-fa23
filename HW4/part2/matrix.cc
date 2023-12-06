#include <mpi.h>
#include <cstdio>
#include <iostream>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

using namespace std;

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    int n, m, l;
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        cin >> n >> m >> l;
        *n_ptr = n;
        *m_ptr = m;
        *l_ptr = l;
        *a_mat_ptr = new int[n * m];
        *b_mat_ptr = new int[m * l];

        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                cin >> (*a_mat_ptr)[i * m + j];
            }
        }

        for (int i = 0; i < m; i++){
            for (int j = 0; j < l; j++){
                cin >> (*b_mat_ptr)[i * l + j];
            }
        }

    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    int rows; // Number of rows of matrix a sent to each worker
    int averow, extra, offset;  // Used to determine rows sent to each worker
    int N, M, L; 
    int *c_mat = new int[n * l]; // Result matrix

    // Master
    if (world_rank == MASTER) {
        // Send matrix data to the worker processes
        averow = n / (world_size - 1);
        extra = n % (world_size - 1);
        offset = 0;

        for (int worker = 1; worker < world_size; worker++){
            rows = (worker <= extra) ? averow + 1 : averow;
            
            MPI_Send(&n, 1, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);

            MPI_Send(&rows, 1, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&offset, 1, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(a_mat + offset * m, rows * m, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b_mat, m * l, MPI_INT, worker, FROM_MASTER, MPI_COMM_WORLD);

            offset += rows;
        }

        // Receive from worker tasks
        for (int worker = 1; worker < world_size; worker++){
            MPI_Recv(&rows, 1, MPI_INT, worker, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&offset, 1, MPI_INT, worker, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(c_mat + offset * l, rows * l, MPI_INT, worker, FROM_WORKER, MPI_COMM_WORLD, &status);
        }

        // Print result
        for (int i = 0; i < n; i++){
            for (int j = 0; j < l; j++) {
                cout << c_mat[i * l + j] << " ";
            }
            cout << endl;
        }

        free(c_mat);
    }

    // Workers
    else if(world_rank > MASTER) {
        MPI_Recv(&N, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&L, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        int *a = new int[N * M];
        int *b = new int[M * L];
        int *c = new int[N * L];

        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(a, rows * M, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(b, M * L, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        // Matrix multiplication
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < L; j++){
                c[i * L + j] = 0;
                for (int k = 0; k < M; k++){
                    c[i * L + j] += a[i * M + k] * b[k * L + j];
                }
            }
        }

        // Send result back to master
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(c, rows * L, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);

        free(a);
        free(b);
        free(c);
    }
    
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        free(a_mat);
        free(b_mat);
    }
}