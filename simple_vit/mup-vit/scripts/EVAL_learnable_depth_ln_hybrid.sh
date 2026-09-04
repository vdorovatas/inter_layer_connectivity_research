#!/bin/bash

#SBATCH --job-name=eval_ldLN_h
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # 32 threads per node
#SBATCH --time=00:30:00               # time limit: 1 hour
#SBATCH --output=outputs/noise/learnable_depthLN_hybrid_025_005.out  # standard output file
#SBATCH --account=EUHPC_A04_051       # account name
#SBATCH --qos=boost_qos_dbg            # temporary partition change
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1

source /leonardo_work/EUHPC_A04_051/vdoro_env/bin/activate

torchrun main.py /leonardo/prod/data/ai/imagenet/ilsvrc2012 --workers 16 --multiprocessing-distributed --evaluate --add_val_noise --val_noise_std 0.3 --epochs 300 --batch-size 1024 --torchvision-inception-crop  --name learnable_depthLN_hybrid_linear-head-300ep_lr1e-3_warmup30k --arch learnable_depthLN_hybrid --lr 0.001 --warmup 30000 --resume /leonardo_work/EUHPC_A04_051/vdoro/simple_vit/mup-vit/logs/learnable_depthLN_hybrid_linear-head-300ep_lr1e-3_warmup30k/checkpoints/model_best.pth.tar


