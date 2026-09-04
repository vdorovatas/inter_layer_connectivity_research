#!/bin/bash

#SBATCH --job-name=LS_skipattn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
##SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/long_short_skipattn_240kiters_08.out
#SBATCH --output=/leonardo_scratch/large/userexternal/edorovat/gpt2/outputs/owt2/base_config/long_short_skipattn_240kiters_08.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=boost_qos_lprod #normal

source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"

export WANDB_ARTIFACTS_DIR="$SCRATCH/wandb_artifacts"
export WANDB_CONFIG_DIR="$SCRATCH/wandb_config"
export WANDB_DATA_DIR="$SCRATCH/wandb_data"
export WANDB_TMPDIR="$SCRATCH/wandb_tmp"

torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 0.8 --alphas_train --skipattn --exp_name "long_short_skipattn_240k_iters_08" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 240000 --n_layer 24 #--wandb True --wandb_project "OWT2" --eval_freq 500 #--l1_regularizer_lamda 0.0001 --binary_regularizer_lamda 0.0001 #--warmup_percent 0.5
