torchrun --nproc_per_node 8 -m training.main \
    --train-data /remote-home/chenyitong/open_clip/openclip_data/coco_train.csv \
    --train-num-samples 965036 \
    --lr 0.01 \
    --warmup 100 \
    --force-quick-gelu \
    --dataset-type csv \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --model  ViT-L-14 \
    --lock-text \
    --lock-image \
    --zeroshot-frequency 1 \
    --save-frequency 20 \
    --epochs 600 \
    --csv-caption-key target \
    --pretrained openai \
    --resume /remote-home/chenyitong/open_clip/src/logs/2023_12_10-13_26_15-model_ViT-L-14-lr_0.01-b_128-j_8-p_amp/checkpoints/epoch_300.pt
    # --voc-val /remote-home/chenyitong/open_clip/openclip_data/voc_dataset.csv \
    # --coco-val /remote-home/chenyitong/open_clip/openclip_data/coco_val.csv