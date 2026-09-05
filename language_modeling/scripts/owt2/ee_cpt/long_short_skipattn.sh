#!/bin/bash

#SBATCH --job-name=LS_skipattn_ee
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/ee_cpt/long_short_skipattn_60kiters.out
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --exp_name "ee_cpt_long_short_skipattn_60k_iters_1" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 40000 --n_layer 24 --ee_training --resume 'exps/owt2/long_short/long_short_skipattn_240k_iters_1/ckpt.pt' --lr 0.0001
