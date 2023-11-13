#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>

using namespace std;

pthread_mutex_t mutexSum;
int num_thread;
long long int total_hit;
long long int num_toss;

// Funtion for every thread
void *calculate_pi(void *args) {
    long long int local_hit = 0;
    long long int num_point = num_toss / num_thread;
    unsigned int seed = 123;

    for (int i = 0; i < num_point; i++){
        double x = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
        double dist = x * x + y * y;

        if (dist <= 1.0)
            local_hit++;
    }
    
    pthread_mutex_lock(&mutexSum);
    total_hit += local_hit;
    pthread_mutex_unlock(&mutexSum);

    return NULL;
}


int main(int argc, char **argv){
    if (argc != 3) {
        printf("Usage: ./pi.out {CPU core} {Number of tosses}\n");
        return 1;
    }

    num_thread = atoi(argv[1]);
    num_toss = atoll(argv[2]);

    pthread_t threads[num_thread];

    // Initialize mutex lock
    pthread_mutex_init(&mutexSum, NULL);

    for (int i = 0; i < num_thread; i++) {
        // Create new thread to run the calculate_pi function with corresponding args[i] arguments
        pthread_create(&threads[i], NULL, calculate_pi, NULL);
    }
    
    /* Wait for the other threads*/
    void *status;
    for (int i = 0;i < num_thread;i++) {
        pthread_join(threads[i], &status);
    }
    
    double pi = 4 * ((total_hit) / (double)num_toss);
    printf("%.7lf\n", pi);

    pthread_mutex_destroy(&mutexSum);

    return 0;
}