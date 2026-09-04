#!/bin/bash

#SBATCH --job-name=hybrid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/hybrid_240kiters_1.out
#SBATCH --account=EUHPC_A04_051
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
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 1.0 --alphas_train --exp_name "hybrid_240k_iters_1" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 240000 --n_layer 24 --wandb True --wandb_project "OWT2" --eval_freq 500 #--warmup_percent 0.5
