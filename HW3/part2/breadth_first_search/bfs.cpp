#include "bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE 1

void vertex_set_clear(vertex_set *list) {
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count) {
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void vertex_set_init_calloc(vertex_set *list, int count) {
    list->max_vertices = count;
    list->vertices = (int *)calloc(list->max_vertices, sizeof(int));
    vertex_set_clear(list);
}

void swap_pointers(vertex_set **set1, vertex_set **set2){
    vertex_set *tmp = *set1;
    *set1 = *set2;
    *set2 = tmp;
}

void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int depth) {
    
    int num_threads = omp_get_max_threads();
    int count_per_thread[num_threads][16] = {0}; // padding 64 bytes to avoid false sharing (assuming cache line is 64 bytes)

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < frontier->count; ++i) {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    distances[outgoing] = depth + 1;
                }
            }
        }

        #pragma omp for
        for (int i = 0; i < g->num_nodes; ++i) {
            if (distances[i] == depth + 1) {
                count_per_thread[thread_id][0]++;
            }
        }

        int ver_idx = 0;
        for (int i = 0; i < thread_id; ++i) {
            ver_idx += count_per_thread[i][0];
        }

        #pragma omp for nowait
        for (int i = 0; i < g->num_nodes; ++i) {
            if (distances[i] == depth + 1) {
                new_frontier->vertices[ver_idx] = i;
                ver_idx++;
            }
        }

        // global count
        #pragma omp atomic
        new_frontier->count += count_per_thread[thread_id][0];
    }
}

void bfs_top_down(Graph graph, solution *sol) {
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    int depth = 0;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, depth);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        swap_pointers(&frontier, &new_frontier);

        depth++;
    }

    free(list1.vertices);
    free(list2.vertices);
}

void bottom_up_step_Hybrid(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int depth) {

    int num_threads = omp_get_max_threads();
    int count_per_thread[num_threads][16] = {0}; // padding 64 bytes to avoid false sharing (assuming cache line is 64 bytes)

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        Vertex *local_frontier;

        local_frontier = (Vertex *)malloc(sizeof(Vertex) * g->num_nodes);

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < g->num_nodes; i++) {
            if (distances[i] == NOT_VISITED_MARKER) {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    if (distances[incoming] == depth) {
                        distances[i] = depth + 1;
                        local_frontier[count_per_thread[thread_id][0]] = i;
                        count_per_thread[thread_id][0]++;
                        break;
                    }
                }
            }
        }

        // Hybrid or debugging mode
        int ver_idx = 0;
        for (int i = 0; i < thread_id; ++i) {
            ver_idx += count_per_thread[i][0];
        }

        for (int i = 0; i < count_per_thread[thread_id][0]; ++i) {
            new_frontier->vertices[ver_idx] = local_frontier[i];
            ver_idx++;
        }

        // global count
        #pragma omp atomic
        new_frontier->count += count_per_thread[thread_id][0];

        free(local_frontier);
    }
}

void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances, int depth, int *change_flag) {

    *change_flag = 0;

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < g->num_nodes; i++) {
        if (distances[i] == NOT_VISITED_MARKER) {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                if (frontier->vertices[incoming]) {
                    new_frontier->vertices[i] = 1;
                    distances[i] = depth;
                    *change_flag = 1;
                    break;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol) {
    vertex_set list1, list2;
    vertex_set_init_calloc(&list1, graph->num_nodes);
    vertex_set_init_calloc(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    int depth = 1;
    int change_flag = 1;

    while (change_flag) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, depth, &change_flag);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        swap_pointers(&frontier, &new_frontier);

        // increment the depth of bfs
        depth++;
    }

    free(list1.vertices);
    free(list2.vertices);
}

void bfs_hybrid(Graph graph, solution *sol) {
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    int depth = 0;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        // Choose TopDown or bottom up        
        if ((float)frontier->count / graph->num_nodes < 0.3)
            top_down_step(graph, frontier, new_frontier, sol->distances, depth);
        else
            bottom_up_step_Hybrid(graph, frontier, new_frontier, sol->distances, depth);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        swap_pointers(&frontier, &new_frontier);

        // increment the depth of bfs
        depth++;
    }

    free(list1.vertices);
    free(list2.vertices);
}

