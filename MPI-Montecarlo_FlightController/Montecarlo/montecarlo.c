#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <time.h>
#include <math.h>

// PCG random number generator (better than rand())
typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;
double pcg32_random(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

    return (double)ran_int / (double)UINT32_MAX;
}


int main(int argc, char** argv) {
    int d;
    long long NUM_SAMPLES = 1000000;
    long long SEED = time(NULL);

    if (argc == 4) {
        d = atoi(argv[1]);
        NUM_SAMPLES = atoll(argv[2]);
        SEED = atoll(argv[3]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;

    double start = MPI_Wtime();
    long long local_count = 0;
    for (int i = 0; i < NUM_SAMPLES / size; i++) {
        double x[d];
        double radius2 = 0.0;
        for (int id = 0; id < d; id++) {
            x[id] = 2.0 * pcg32_random(&rng) - 1.0;
            radius2 += x[id] * x[id];
        }

        if (radius2 <= 1.0)
            local_count++;
    }

    long long global_count;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    double elapsed_time = end - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        //double ratio = (double)global_count / NUM_SAMPLES;
        double ratio = exp(log((double)global_count) - log((double)NUM_SAMPLES));
        double theo_ratio = pow(M_PI, 0.5*d) / tgamma(0.5*d+1.0) / pow(2.0, d);
        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %lld samples, d: %d, seed %lld, size: %d \n", NUM_SAMPLES, d, SEED, size);
        printf("Ratio = %.3e (%.3e) Err: %.2e\n", ratio, theo_ratio, fabs(ratio-theo_ratio)/theo_ratio);
        printf("Elapsed time: %.3f seconds\n", elapsed_time);
    }

    MPI_Finalize();
    return 0;
}