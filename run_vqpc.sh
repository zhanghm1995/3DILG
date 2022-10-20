set -x

torchrun --nproc_per_node=1 --master_port=20002 run_vqpc.py \
         --output_dir output/vqpc_512_1024_2048_debug \
         --log_dir output/vqpc_512_1024_2048_debug/logs/ \
         --model vqpc_512_1024_2048 --data_path ./data/train/PUGAN_poisson_256_poisson_1024.h5 \
         --batch_size 32 --num_workers 6 --lr 1e-3 \
         --point_cloud_size 1024  \
         --save_ckpt_freq 10 --validation_freq 1 \
         --save_ckpt
