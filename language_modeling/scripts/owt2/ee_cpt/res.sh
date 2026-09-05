#!/bin/bash

#SBATCH --job-name=res_ee
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/ee_cpt/res_60kiters.out
#SBATCH --account=EUHPC_A04_051
#SBATCH --qos=boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "ee_cpt_res_60k_iters" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 40000 --n_layer 24 --eval_freq 500 --ee_training --resume 'exps/owt2/hybrid/res_240k_iters/ckpt.pt' --lr 0.0001
