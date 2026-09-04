#!/bin/bash

#SBATCH --job-name=LSN_CL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/continual_learning/C1/lsn_skipattn_1.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=normal


source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

LR=0.00005
BS=32
ACC_STEPS=1
# TASK 1
torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --exp_name "CL11_boolq_lsn_6k_iters" --dataset 'boolq' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 1400 --n_layer 24 --eval_freq 500 --resume 'exps/owt2/long_short/long_short_skipattn_240k_iters_1/ckpt.pt' --lr $LR  --acc_steps $ACC_STEPS --batch_size $BS

# TASK 2
torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --exp_name "CL12_hellaswag_lsn_6k_iters" --dataset 'hellaswag' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 6250 --n_layer 24 --eval_freq 500 --resume 'exps/boolq/long_short/CL11_boolq_lsn_6k_iters/ckpt.pt' --lr $LR  --acc_steps $ACC_STEPS --batch_size $BS

# TASK 3
torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --exp_name "CL13_piqa_lsn_2k_iters" --dataset 'piqa' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 2500 --n_layer 24 --eval_freq 500 --resume 'exps/hellaswag/long_short/CL12_hellaswag_lsn_6k_iters/ckpt.pt' --lr $LR  --acc_steps $ACC_STEPS --batch_size $BS

# TASK 4
torchrun --nproc_per_node=4 main.py --model 'long_short' --alphas_mean 1.0 --alphas_train --skipattn --exp_name "CL14_arc_easy_lsn_3k_iters" --dataset 'arc_easy' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations 400 --n_layer 24 --eval_freq 500 --resume 'exps/piqa/long_short/CL13_piqa_lsn_2k_iters/ckpt.pt' --lr $LR  --acc_steps $ACC_STEPS --batch_size $BS
