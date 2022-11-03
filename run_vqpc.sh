#!/bin/bash
#SBATCH -J s1_full_512GT
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_s1_full_train_data_512_GT/%j.out
#SBATCH -e /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_s1_full_train_data_512_GT/%j.out


export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run_vqpc.py \
         --epochs 100 --lr 1e-3 --min_lr 1e-4 --warmup_epochs 0 \
         --output_dir output/vqpc_s1_full_train_data_512_GT \
         --log_dir output/vqpc_s1_full_train_data_512_GT/logs/ \
         --model vqpc_256_1024_1024 --data_path ./data/train/PUGAN_poisson_256_poisson_1024.h5 \
         --batch_size 32 --num_workers 6 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --save_ckpt --stage 'stage1' --disable_eval \
#         --test --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_256_1024_1024_refine_100_epoch_more_losses/checkpoint-79.pth"
