#!/bin/bash

#SBATCH --job-name=streaming_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuloq
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH --output=outputs/dummy.out 
#SBATCH --mail-user=vaggelis.dorovatas@toyota-europe.com
#SBATCH --nodelist=gpu009

python  dummy.py
