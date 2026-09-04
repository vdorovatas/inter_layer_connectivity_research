#!/bin/bash

#SBATCH --job-name=linear_paths_double_pass_res
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         	# 32 threads per node
#SBATCH --time=18:00:00               # time limit: 1 hour
#SBATCH --output=outputs/dg_res_linear_paths_scale_grads.out   
#SBATCH --account=EUHPC_D33_268 # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u linear_paths_double_pass_main.py --arch dfa_res --epochs 300
