#!/bin/bash

#SBATCH --job-name=vit_deb
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=2:00:00               # time limit: 1 hour
#SBATCH --output=outputs/dum.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python scripts/dum.py


