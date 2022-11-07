#!/bin/bash
#SBATCH -J s2_linear
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage2_random_pointnet_k_32_test_4096/%j.out
#SBATCH -e /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage2_random_pointnet_k_32_test_4096/%j.out


export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run_vqpc.py \
         --epochs 100 --lr 1e-3 --min_lr 5e-4 --warmup_epochs 0 \
         --output_dir output/vqpc_stage2_random_pointnet_k_32_test_4096 \
         --log_dir output/vqpc_stage2_random_pointnet_k_32_test_4096/logs/ \
         --model vqpc_stage2 --data_path ./data/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5 \
         --batch_size 64 --num_workers 6 --lr 1e-3 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --save_ckpt  --stage 'stage2' --disable_eval \
         --test_set_size 'PUGAN_input_2048' --num_test_GT_points 8192 \
         --test --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage2_random_pointnet/checkpoint-99.pth"
