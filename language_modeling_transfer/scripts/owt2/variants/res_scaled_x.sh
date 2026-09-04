#!/bin/bash

#SBATCH --job-name=res_x
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/res_scaled_x_init_0_240k_iters.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'res_scaled_x' --exp_name "init_0_240k_iters" --dataset 'owt2' --save_checkpoint_freq 100000000000 --iterations 240000 --n_layer 24
