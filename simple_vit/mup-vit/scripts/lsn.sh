#!/bin/bash

#SBATCH --job-name=LS-ViT
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=76:00:00               # time limit: 1 hour
#SBATCH --output=/leonardo_scratch/large/userexternal/edorovat/simple_vit/outputs/final_LSN_300ep_depthLN_clamp_08.out 
#SBATCH --account=EUHPC_D33_268       # account name
#SBATCH --qos=boost_qos_lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 300 --batch-size 1024 --torchvision-inception-crop  --name final_lsn_300ep_depthLN_clamp_08 --arch long_short --lr 0.001 --warmup 30000 --depthLN #--mlp-head #--resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/lsn_500ep/checkpoints/checkpoint.pth.tar


