#!/bin/bash

#SBATCH --job-name=ac_vit
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=96:00:00               # time limit: 1 hour
#SBATCH --output=outputs/acn_dirac_ep700.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

python3 -u main.py --lr=0.0001 --batch_size=256 --weight_decay=0.01 --mode='train' --model_type long --data_path /leonardo/prod/data/ai/imagenet/ilsvrc2012 --epochs 700  #--resume True

