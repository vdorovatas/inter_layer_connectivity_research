#!/bin/bash

#SBATCH --job-name=res_CL_ood
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/outputs/continual_learning/OOD_C3/res.out
#SBATCH --account=EUHPC_D33_268
#SBATCH --qos=normal

source /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments/env/bin/activate
cd /leonardo_work/EUHPC_A04_051/vdoro/language_modeling/DenseFormer/experiments

LR=0.00005
BS=32
ACC_STEPS=1
MEDQA_STEPS=3120
BIOLOGY_STEPS=1060
CHEMISTRY_STEPS=1250
# TASK 1
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "OOD_CL31_chemistry_res" --dataset 'chemistry' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations $CHEMISTRY_STEPS --n_layer 24 --eval_freq 200 --resume 'exps/owt2/hybrid/res_240k_iters/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS

# TASK 2
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "OOD_CL32_medqa_res" --dataset 'medqa' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations $MEDQA_STEPS --n_layer 24 --eval_freq 200 --resume '/leonardo_scratch/large/userexternal/edorovat/gpt2/exps/chemistry/hybrid/OOD_CL31_chemistry_res/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS

# TASK 3
torchrun --nproc_per_node=4 main.py --model 'hybrid' --alphas_mean 0.0 --exp_name "OOD_CL33_biology_res" --dataset 'biology' --save_checkpoint_freq 1000000000 --grad_clip 1.0 --iterations $BIOLOGY_STEPS --n_layer 24 --eval_freq 200 --resume '/leonardo_scratch/large/userexternal/edorovat/gpt2/exps/medqa/hybrid/OOD_CL32_medqa_res/ckpt.pt' --lr $LR --acc_steps $ACC_STEPS --batch_size $BS
