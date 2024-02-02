CUDA_VISIBLE_DEVICES=0 python train_vit_domain_genralization.py \
  --data_dir /opt/fumh/datasets/DG \
  --load_path /path/to/Vit-B_16.npz \
  --source_dataset imagenet \
  --target_dataset imagenet \
  --model vit_base_patch16_224_in21k \
  --batch_size 32 \
  --batch_size_test 256 \
  --epochs 100 \
  --warmup_epochs 10 \
  --fusion_size 3 \
  --r 2 \
  --beta 100.0 \
  --lora_before_blocks 0-11 \
  --add_after_blocks 6-11 \
  --weight_decay 0.05 \
  --lr 0.0001 \
  --drop_path 0.0 \
  --amp \
  --prefetcher

