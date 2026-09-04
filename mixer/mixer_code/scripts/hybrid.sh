#!/bin/bash

#SBATCH --job-name=hybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         	# 32 threads per node
#SBATCH --time=12:00:00               # time limit: 1 hour
#SBATCH --output=new_outputs/ldLN_hybrid_300ep_025_005_run2.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u main.py --arch hybrid --epochs 300
