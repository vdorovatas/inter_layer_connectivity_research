#!/bin/bash

#SBATCH --job-name=l_dLN_hybrid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/learnable_depthLN_hybrid_0_240k_iters.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'learnable_depthLN_hybrid' --dataset 'owt2' --exp_name "init_0_240k_iters_run2" --alphas_mean 0.0 --alphas_std 0.0 --save_checkpoint_freq 50000000000 --grad_clip 1.0 --iterations 240000 --n_layer 24
