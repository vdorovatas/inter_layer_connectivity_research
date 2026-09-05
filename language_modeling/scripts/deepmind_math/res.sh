#!/bin/bash

#SBATCH --job-name=Res_deepmath
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/deepmind_math/res_from_pretrained_30kiters.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=normal #boost_qos_lprod #normal


source $WORK/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

torchrun --nproc_per_node=4 main_deepmind_math.py --model 'hybrid' --alphas_mean 0.0 --exp_name "res_skipattn_from_pretrained_30k_iters" --dataset 'deepmind_math'  --mask_questions --pack_sequences --save_checkpoint_freq 100000000000 --grad_clip 1.0 --iterations 30000 --n_layer 24 --eval_freq 500 --resume 'exps/owt2/hybrid/res_240k_iters/ckpt.pt' --lr 0.00005 
