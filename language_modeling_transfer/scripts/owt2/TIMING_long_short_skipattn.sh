#!/bin/bash

#SBATCH --job-name=LS_skipattn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/TIMING_trainable_long_short_skipattn_240kiters_1.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"

export WANDB_ARTIFACTS_DIR="$SCRATCH/wandb_artifacts"
export WANDB_CONFIG_DIR="$SCRATCH/wandb_config"
export WANDB_DATA_DIR="$SCRATCH/wandb_data"
export WANDB_TMPDIR="$SCRATCH/wandb_tmp"

torchrun --nproc_per_node=4 main.py --model 'long_short' --skipattn --alphas_train --exp_name "TIMING_trainable_long_short_skipattn_240k_iters_1" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 240000 --n_layer 24 --alphas_init_list 0.043 0.051 0.054 0.299 0.022 0.025 0.026 0.000 0.110 0.107 0.638 0.067 0.076 0.079 0.083 0.001 0.102 0.102 0.124 0.171 0.207 0.260 0.352 0.541 0.536
