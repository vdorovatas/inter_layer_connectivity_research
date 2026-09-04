#!/bin/bash

#SBATCH --job-name=rep
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         	# 32 threads per node
#SBATCH --time=1:00:00               # time limit: 1 hour
#SBATCH --output=new_outputs/analyze_representations.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u analyze_main.py #--arch acn #--checkpoint inherent_EE_checkpoints/res_480.pth.tar
