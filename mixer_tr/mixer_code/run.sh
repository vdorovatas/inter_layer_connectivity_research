#!/bin/bash

#SBATCH --job-name=eval2
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuloq
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED 
##SBATCH --nodelist=gpu007
##SBATCH --exclusive
#SBATCH --output=outputs_mix/lcn_c100.out 
##SBATCH --mail-type=ALL
##SBATCH --mail-user=vaggelis.dorovatas@toyota-europe.com 

python main_long.py 
