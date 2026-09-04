#!/bin/bash

#SBATCH --job-name=hybrid
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 32 threads per node
#SBATCH --time=12:00:00               # time limit: 1 hour
#SBATCH --output=outputs/cifar10/hybrid_025_005.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=normal #boost_qos_dbg #lprod            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun cifar10_main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --epochs 15 --batch-size 128 --torchvision-inception-crop  --name hybrid_300ep --arch hybrid --lr 0.0005 --warmup 800 --resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/hybrid_300ep/checkpoints/model_best.pth.tar 


