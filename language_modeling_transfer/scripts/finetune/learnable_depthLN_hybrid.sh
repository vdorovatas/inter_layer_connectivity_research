#!/bin/bash

#SBATCH --job-name=l_dLN_hybrid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/finetune/learnable_depthLN_hybrid_025_005_240k_iters.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'learnable_depthLN_hybrid' --dataset 'hellaswag' --exp_name "init_025_005_240k_iters" --alphas_mean 0.25 --alphas_std 0.005 --save_checkpoint_freq 50000000000 --grad_clip 1.0 --iterations 1000 --n_layer 24 --warmup_percent 0.05 --lr 0.0005 --eval_freq 25
