#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void initialize(double *v, int N) {
	for (int i = 0; i < N; i++) {
		v[i] = (1 - pow(0.5 - (double)i/(double)N, 2)) * cos(2*M_PI*100* (i - 0.5)/N);
	}
}

void argmax_par(double *v, int N, double *m, int *idx_m) {

	*m = -1.;
	*idx_m = -1;

  	#pragma omp parallel for shared(m, idx_m)
	for (int i = 0; i < N; i++) {
  		#pragma omp critical
		if (v[i] > *m) {
			*m = v[i];
			*idx_m = i;
		}
	}
}


void argmax_seq(double *v, int N, double *m, int *idx_m) {

	*m = -1.;
	*idx_m = -1;

	for (int i = 0; i < N; i++) {
		if (v[i] > *m) {
			*m = v[i];
			*idx_m = i;
		}
	}
}

void argmax_recursive(double *v, int N, double *m, int *idx_m, int K) {

	if (N <= K) {
		argmax_seq(v, N, m, idx_m);
		return;
	}

	double ml, mr;
	int idx_ml, idx_mr;

	argmax_recursive(v, N/2, &ml, &idx_ml, K);
	argmax_recursive(v + N/2, N/2, &mr, &idx_mr, K);

	if (mr >= ml) {
		*m = mr;
		*idx_m = idx_mr + N/2;
	}
	else {
		*m = ml;
		*idx_m = idx_ml;
	}
	return;
}

void argmax_recursive_tasks(double *v, int N, double *m, int *idx_m, int K) {

	if (N <= K) {
		argmax_seq(v, N, m, idx_m);
		return;
	}

	double ml, mr;
	int idx_ml, idx_mr;

	#pragma omp task shared(ml, idx_ml) 
	argmax_recursive_tasks(v, N/2, &ml, &idx_ml, K);

	#pragma omp task shared(mr, idx_mr)
	argmax_recursive_tasks(v + N/2, N/2, &mr, &idx_mr, K);

	#pragma omp taskwait

	if (mr >= ml) {
		*m = mr;
		*idx_m = idx_mr + N/2;
	}
	else {
		*m = ml;
		*idx_m = idx_ml;
	}
	return;
}

int main(int argc, char * argv[])
{
 	int N = 4096*4096;
	int K = (argc == 2) ? atoi(argv[1]) : 512;

	printf("Running argmax with K = %d\n", K);

	double *v = (double *)malloc(N*sizeof(double));

	double t_start, t_end;

	initialize(v, N);

	t_start = omp_get_wtime();
	double seq_m;
	int seq_idx_m;
  	argmax_seq(v, N, &seq_m, &seq_idx_m);
	t_end = omp_get_wtime();
	printf("sequential for       argmax: m = %5.2f, idx=%d, time=%fs\n", seq_m, seq_idx_m, t_end - t_start);

	t_start = omp_get_wtime();
	double par_m;
	int par_idx_m;
  	argmax_par(v, N, &par_m, &par_idx_m);
	t_end = omp_get_wtime();
	printf("parallel   for       argmax: m = %5.2f, idx=%d, time=%fs\n", par_m, par_idx_m, t_end - t_start);

	t_start = omp_get_wtime();
	double rec_m;
	int rec_idx_m;
  	argmax_recursive(v, N, &rec_m, &rec_idx_m, K);
	t_end = omp_get_wtime();
	printf("sequential recursive argmax: m = %5.2f, idx=%d, time=%fs\n", rec_m, rec_idx_m, t_end - t_start);

	t_start = omp_get_wtime();
	double task_m;
	int task_idx_m;
	#pragma omp parallel
	#pragma omp single
  	argmax_recursive_tasks(v, N, &task_m, &task_idx_m, K);
	t_end = omp_get_wtime();
	printf("parallel   recursive argmax: m = %5.2f, idx=%d, time=%fs\n", task_m, task_idx_m, t_end - t_start);

	free(v);
	return 0;
}
