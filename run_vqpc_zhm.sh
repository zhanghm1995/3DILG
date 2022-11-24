#!/bin/bash
#SBATCH -J S1_hd1
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage1_random_2048_codes_only_fine_loss_resume_e17_cd_100_hd_1_test/%j.out
#SBATCH -e /mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage1_random_2048_codes_only_fine_loss_resume_e17_cd_100_hd_1_test/%j.out


export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run_vqpc.py \
         --epochs 100 --lr 1e-3 --min_lr 1e-5 --warmup_epochs 0 \
         --output_dir output/vqpc_stage1_random_2048_codes_only_fine_loss_normalize_patch_wo_scale_aug \
         --log_dir output/vqpc_stage1_random_2048_codes_only_fine_loss_normalize_patch_wo_scale_aug/logs/ \
         --model vqpc_256_2048_1024 \
         --data_path ./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5 \
         --batch_size 64 --num_workers 6 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --save_ckpt --stage 'stage1' --disable_eval \
         --test_set_size 'PU1K_input_2048' --num_test_GT_points 8192
        #  --test --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/vqpc_stage1_random_2048_codes_only_fine_loss_resume_e17_cd_100_hd_1/best_cd.pth" \
