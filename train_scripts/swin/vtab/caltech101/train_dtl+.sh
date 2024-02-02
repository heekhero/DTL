CUDA_VISIBLE_DEVICES=0 python train_swin_vtab.py \
  --data_dir  /path/to/VTAB/vtab-1k \
  --dataset caltech101 \
  --model swin_base_patch4_window7_224_in22k \
  --batch_size 32 \
  --batch_size_test 256 \
  --epochs 100 \
  --warmup_epochs 10 \
  --fusion_size 5 \
  --r 4 \
  --beta 100.0 \
  --lora_before_blocks 4-23 \
  --add_after_blocks 14-23 \
  --weight_decay 0.05 \
  --lr 0.0005 \
  --drop_path 0.4 \
  --amp \
  --prefetcher
