#!/bin/bash

#SBATCH --job-name=dformer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/pg19/base_config/denseformer_acn_init_240kiters.out
##SBATCH --error=/leonardo_work/EUHPC_A04_051/babylm/evaluation-pipeline-2025/outputs/eval-blimp-maskedlm-hybrid.err
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'denseformer' --exp_name "acn_init_240k_iters"  --dataset 'pg19' --save_checkpoint_freq 100000000000 --iterations 240000 --grad_clip 1.0 --eval_freq 500
