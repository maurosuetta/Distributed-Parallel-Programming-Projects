#!/bin/bash
#SBATCH --job-name=fc_mpi_job
#SBATCH --output=fc_mpi_%j.out
#SBATCH --error=fc_mpi_%j.err
#SBATCH --partition=std
#SBATCH --ntasks=80
#SBATCH --time=00:05:00

module load gcc/13.3.0
module load openmpi/5.0.3

make || exit 1

srun ./fc_mpi input_planes_10kk.txt 25 2 0


