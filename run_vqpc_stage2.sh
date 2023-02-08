#!/bin/bash
#SBATCH -J 10seed_better
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o /mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_CE_pretrain_vq_match_lr_points_CE_mean_possion_input_cal_train_params/%j.out
#SBATCH -e /mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_CE_pretrain_vq_match_lr_points_CE_mean_possion_input_cal_train_params/%j.out


export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT run_vqpc.py \
         --epochs 100 --lr 1e-3 --min_lr 1e-5 --warmup_epochs 0  \
         --output_dir output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_CE_pretrain_vq_match_lr_points_CE_mean_possion_input_cal_train_params \
         --log_dir output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_CE_pretrain_vq_match_lr_points_CE_mean_possion_input_cal_train_params/logs/ \
         --model vqpc_stage2_new_prediction_head --data_path ./data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5\
         --batch_size 64 --num_workers 6 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --stage 'stage2' --disable_eval \
         --test_set_size 'PUGAN_input_1024x4' --num_test_GT_points 4096 \
#         --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_pretrain_vq_match_lr_points_possion_input_ablation_NN/best_cd.pth" #"/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_FPS_4_pe_EMA_VQ_1024_fix_decoder_CE_pretrain_vq_match_lr_points_CE_mean/best_cd.pth"
#         --test --resume "/mntnfs/cui_data4/yanchengwang/3DILG/output/stage2_random_4_pe_EMA_VQ_1024_not_fix_01_pretrain_loss_not_fix_decoder/best_cd.pth" --clip_grad 1
#pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5 \
