#!/bin/bash
#SBATCH --job-name=amax
#SBATCH	-p std
#SBATCH	--output=job.out
#SBATCH	--error=job.err
#SBATCH	--ntasks=1
#SBATCH	--cpus-per-task=8
#SBATCH	--time=00:00:10

#lscpu

export OMP_NUM_THREADS=4

make
./argmax 1
./argmax 128
./argmax 512
./argmax 1024
./argmax 2048

