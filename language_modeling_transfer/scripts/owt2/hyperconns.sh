#!/bin/bash

#SBATCH --job-name=hc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
##SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/owt2/base_config/hyperconns_240kiters.out
#SBATCH --output=/leonardo_scratch/large/userexternal/edorovat/gpt2/outputs/owt2/base_config/hyperconns_240kiters.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=boost_qos_lprod #normal


source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main.py --model 'hc' --hc_num_streams 2 --exp_name "240k_iters" --dataset 'owt2' --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 240000 --n_layer 24 --eval_freq 500 #--batch_size 64 --acc_steps 8 
