#!/bin/bash

#SBATCH --job-name=vit_acn100
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=24:00:00               # time limit: 1 hour
#SBATCH --output=outputs/acn_100epochs.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 100 --batch-size 1024 --torchvision-inception-crop  --name linear-head-100ep --arch acn --lr 0.0005 --warmup 50000


