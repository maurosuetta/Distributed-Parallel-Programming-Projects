#!/bin/bash

#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_seq_%j.out
#SBATCH --error=out_seq_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

### module load conda
### conda create -n image_plotter
### conda activate image_plotter
### conda install matplotlib opencv
### python plot.py

make >> make.out || exit 1     

./partis_seq 50000 0

