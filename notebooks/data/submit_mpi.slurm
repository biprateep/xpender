#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu
#SBATCH --time 360
#SBATCH --qos=regular
#SBATCH --image=biprateep/desi-dask:latest
#SBATCH --account=desi

srun shifter python create_cutouts_mpi.py

