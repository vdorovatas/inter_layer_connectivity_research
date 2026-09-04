#!/bin/bash

#SBATCH --job-name=chybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         	# 32 threads per node
#SBATCH --time=24:00:00               # time limit: 1 hour
#SBATCH --output=new_outputs/clamped_hybrid_300ep_2classes.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u main.py --arch clamped_hybrid --epochs 300
