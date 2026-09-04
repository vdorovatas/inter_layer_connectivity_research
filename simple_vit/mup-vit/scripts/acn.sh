#!/bin/bash

#SBATCH --job-name=acn
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=96:00:00               # time limit: 1 hour
#SBATCH --output=outputs/acn_700epochs.out           # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 700 --batch-size 1024 --torchvision-inception-crop  --name acn_700ep --arch acn --lr 0.001 --warmup 60000 #--resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/acn_linear-head-700ep_lr1e-3_warmup80k/checkpoints/checkpoint.pth.tar
