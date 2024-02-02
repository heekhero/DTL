CUDA_VISIBLE_DEVICES=0 python train_vit_few_shot.py \
  --data_dir  /path/to/FGFS \
  --load_path /path/to/Vit-B_16.npz \
  --dataset fgvc-aircraft-FS \
  --model vit_base_patch16_224_in21k \
  --batch_size 32 \
  --batch_size_test 256 \
  --epochs 100 \
  --warmup_epochs 10 \
  --fusion_size 0 \
  --r 2 \
  --beta 100.0 \
  --lora_before_blocks 0-11 \
  --add_after_blocks 6-11 \
  --weight_decay 0.05 \
  --shot 4 \
  --lr 0.001 \
  --drop_path 0.0 \
  --amp \
  --prefetcher

