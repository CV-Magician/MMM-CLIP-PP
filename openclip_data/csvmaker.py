import tqdm
import os
import pandas as pd
import json

file_root = '/remote-home/share/coco/train2014/'
coco_clsname = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

with open("./train.json", "r", encoding="utf-8") as f:
    content = json.load(f)
    print(len(coco_clsname))

    data = []
    for index in range(len(content)):
        target = [int(content[index][i]) for i in coco_clsname]
        filepath = file_root + content[index]['name']
        data.append([filepath, target])
        # print(filepath, target)
        # break



# csvfile_path = '/remote-home/chenyitong/PJ_original_dataset/multilabel_modified/multilabel_classification(6)-reduced_modified.csv'
# rd_df = pd.read_csv(csvfile_path)


    
df = pd.DataFrame(data=data)
df.to_csv('coco_train.csv', sep='\t', header=['filepath', 'target'], index=False, mode='a')