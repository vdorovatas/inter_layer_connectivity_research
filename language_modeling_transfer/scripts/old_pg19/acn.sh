#!/bin/bash

#SBATCH --job-name=acn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/pg19/base_config/acn.out
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'acn' --exp_name "base" --lr 0.001 --iterations 200000 --save_checkpoint_freq 100000000000 --n_layer 12 --scheduler linear --grad_clip 1.0
