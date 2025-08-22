#!/bin/bash
# Configuration for 1 node, 4 cores and 5 minutes of execution time
#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_weak_%j.out
#SBATCH --error=out_weak_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=192
#SBATCH --nodes=1
#SBATCH --time=00:05:00

module load gcc/13.3.0
module load openmpi/4.1.1

make >> make.out || exit 1      # Exit if make fails

d=2
mpirun -n 192 ./montecarlo $d 19200000000 1744318604
mpirun -n 64 ./montecarlo $d 6400000000 1744318604
mpirun -n 16 ./montecarlo $d 1600000000 1744318604
mpirun -n 4 ./montecarlo  $d 400000000 1744318604
mpirun -n 1 ./montecarlo  $d 100000000 1744318604

d=3
mpirun -n 192 ./montecarlo $d 19200000000 1744318604
mpirun -n 64 ./montecarlo $d 6400000000 1744318604
mpirun -n 16 ./montecarlo $d 1600000000 1744318604
mpirun -n 4 ./montecarlo  $d 400000000 1744318604
mpirun -n 1 ./montecarlo  $d 100000000 1744318604

d=10
mpirun -n 192 ./montecarlo $d 19200000000 1744318604
mpirun -n 64 ./montecarlo $d 6400000000 1744318604
mpirun -n 16 ./montecarlo $d 1600000000 1744318604
mpirun -n 4 ./montecarlo  $d 400000000 1744318604
mpirun -n 1 ./montecarlo  $d 100000000 1744318604
