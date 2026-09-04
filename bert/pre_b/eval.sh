#!/bin/bash

#SBATCH --job-name=pre_v
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuloq
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED 
#SBATCH --nodelist=gpu009
##SBATCH --exclusive
#SBATCH --output=outputs/output.out 
##SBATCH --mail-type=ALL
##SBATCH --mail-user=vaggelis.dorovatas@toyota-europe.com 

python eval.py --lcn True
#accelerate launch --multi_gpu main.py