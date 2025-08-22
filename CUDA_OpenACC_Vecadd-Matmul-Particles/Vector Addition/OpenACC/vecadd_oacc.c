#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void vecadd_seq(double *A, double *B, double *C, const int N)
{
    #pragma acc parallel loop copyin(A[:N], B[:N]) copyout(C[:N]) firstprivate(N)
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    int N;

    if (argc != 2)
    {
        printf("Usage: %s <vector size N>\n", argv[0]);
        return 1;
    }
    else
    {
        N = atoi(argv[1]);
    }
    printf("Vector size: %d\n", N);

    //
    // Memory allocation
    //
    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));

    //
    // Create random values
    //
    for (int i = 0; i < N; i++)
    {
        A[i] = (double) i;
        B[i] = (double) (2 * (N - i));
        C[i] = -1.0;
    }

    //
    // Vector addition
    //
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vecadd_seq(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    //
    // Validation
    //
    for (int i = 0; i < N; i++)
    {
        double expected = (double) 2 * N - i;
        double local_err = fabs(expected - C[i]);
        if (local_err > 1.0e-6)
        {
            printf("Error at i = %d: fabs( %f - c[%d] ) = %e > %e\n", i, expected, i, local_err, 1.0e-6);
            return 1;
        }
    }

    //
    // Free memory
    //
    free(A);
    free(B);
    free(C);

    return 0;
}