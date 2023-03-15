# -*- encoding: utf-8 -*-
'''
@File    :   convertLabelTool2COCO.py
@Time    :   2023/03/03 10:10:28
@Author  :   Gang Xu 
@Version :   1.0
@Contact :   xg9712@arcsoft.com.cn
@Desc    :   None
'''

import os
import argparse
from pathlib import Path
import random
random.seed(0)
import cv2
import numpy as np
import json

CLASSES = [
    'thick_cylinder', 
    'thin_cylinder', 
    'triangle', 
    'special_car',
    'truck_box',
    'motortricycle',
    'tricycle',
    'biking_person',
    'car',
    'truck_nobox_full',
    'bus',
    'ride_bicycle_person',
    'walking_person',
    'other']

CATEGORIES = []
for idx, cls in enumerate(CLASSES):
    CATEGORIES.append(
        dict(
        id=idx+1,
        name=cls
        )
    )

def dumpAnno(img_paths, outfile):
    ojb_count = 0
    annotations = []
    images = []

    for idx in range(len(img_paths)):
        height, width = cv2.imread(str(img_paths[idx])).shape[:2]
        images.append(dict(
            id=idx,
            width=width,
            height=height,
            file_name=str(img_paths[idx])))
        label_file = img_paths[idx].with_suffix('.txt')
        lines = open(label_file, 'r').readlines()
        for line in lines:
            label = json.loads(line.strip())
            cls = label.get('shape', None) or label.get('nature', None)
            if cls is None:
                if 'vehicle_ped' in str(label_file.parent):
                    cls = 'car'
                else:
                    cls = 'other'
            pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y = list(map(int,label['rect'].strip().split(',')))
            x_min = int(np.clip(min(pt1x, pt2x, pt3x, pt4x), 0, width-1))
            x_max = int(np.clip(max(pt1x, pt2x, pt3x, pt4x), 0, width-1))
            y_min = int(np.clip(min(pt1y, pt2y, pt3y, pt4y), 0, height-1))
            y_max = int(np.clip(max(pt1y, pt2y, pt3y, pt4y), 0, height-1))
            annotations.append(dict(
                id=ojb_count,
                image_id=idx,
                category_id=CLASSES.index(cls)+1,
                bbox=[x_min, y_min, x_max-x_min, y_max- y_min],
                area = int((x_max-x_min) *(y_max-y_min)),
                iscrowd=0
            ))
            ojb_count+=1
    with open(outfile, 'w', encoding='utf8') as f:
        f.write(json.dumps(dict(images=images,annotations=annotations,categories=CATEGORIES),ensure_ascii=False))  


def cvtLabelTool2COCO(root):
    if isinstance(root, str):
        root = Path(root)

    train_outfile = root.joinpath('train_anno.json')
    val_outfile = root.joinpath('val_anno.json')
    folders = [x for x in root.rglob("*") if x.is_dir and x.name=="remapped"]
    img_paths = []
    listOfpaths = [[x for x in d.iterdir() if x.suffix!='.txt'] for d in folders]
    for listOfpath in listOfpaths:
        img_paths.extend(listOfpath)
    img_paths = sorted(img_paths)
    num_imgs = len(img_paths)
    num_train = int(0.95*num_imgs)
    num_val = num_imgs - num_train
    print(f'Found total {num_imgs} images, {num_train} as train, {num_val} as val.')
    random.shuffle(img_paths)
    dumpAnno(img_paths[:num_train], train_outfile)
    dumpAnno(img_paths[num_train:], val_outfile)

def get_args(arglist=None):
    parser = argparse.ArgumentParser('None')
    parser.add_argument('root')
    return parser.parse_args() if arglist is None else parser.parse_args(arglist)

if __name__ == '__main__':
    arglist = [
        '/mnt/104sdf/apaObjectDetection/haomo/vehicle_ped'
    ]
    args = get_args(arglist)
    cvtLabelTool2COCO(args.root)