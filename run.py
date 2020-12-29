import torch
from torchvision import transforms
import json
import matplotlib.pyplot as plt
import random
from dataset.coco import CocoDataset,category_id_to_name
from visualization.vis import visualize
from dataset.aug import transform


coco = CocoDataset("/Users/zhifei/02_data/COCO/minicoco2014","/Users/zhifei/02_data/COCO/mini_coco_val.json")

for i in range(len(coco)):
    image,annos = coco[i]['img'],coco[i]['annot']
    bboxes = annos[:,:4]
    category_ids = annos[:,4]+1
    visualize(image, bboxes, category_ids, category_id_to_name)
    random.seed(7)
    for j in range(5):
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        visualize(
            transformed['image'],
            transformed['bboxes'],
            transformed['category_ids'],
            category_id_to_name,
        )
    break