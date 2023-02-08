#!/bin/bash
#SBATCH -J autoencoder
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_autoencoder_random_4_pe_EMA_VQ_use_1024_codes_FPS_4/%j.out
#SBATCH -e /mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_autoencoder_random_4_pe_EMA_VQ_use_1024_codes_FPS_4/%j.out


export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run_vqpc.py \
         --epochs 100 --lr 1e-3 --min_lr 1e-5 --warmup_epochs 0 \
         --output_dir output/stage1_autoencoder_random_4_pe_EMA_VQ_use_1024_codes_FPS_4 \
         --log_dir output/stage1_autoencoder_random_4_pe_EMA_VQ_use_1024_codes_FPS_4/logs/ \
         --model stage1_autoencoder \
         --data_path ./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5 \
         --batch_size 64 --num_workers 6 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --save_ckpt --stage 'autoencoder' --disable_eval \
         --test_set_size 'PU1K_input_2048' --num_test_GT_points 8192 \
#         --test --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_random_4_pe_EMA_VQ_use_pretrained_feature_as_GT_1024_codes_not_fix_resume/best_cd.pth" \
#         --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage1_random_4_pe_EMA_VQ_use_pretrained_feature_as_GT_1024_codes_not_fix/best_cd.pth" \
