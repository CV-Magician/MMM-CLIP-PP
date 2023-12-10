python -m training.main \
    --model  ViT-L-14 \
    --pretrained openai \
    --dataset-type csv \
    --batch-size 256 \
    --precision amp \
    --zeroshot-frequency 1 \
    --csv-caption-key target \
    --coco-val /remote-home/chenyitong/open_clip/openclip_data/coco_val.csv \
    --resume /remote-home/chenyitong/open_clip/src/logs/2023_12_09-13_16_31-model_ViT-L-14-lr_0.005-b_256-j_8-p_amp/checkpoints/epoch_500.pt