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
from tqdm import tqdm
import mmcv

# CLASSES = [
#     'thick_cylinder', 
#     'thin_cylinder', 
#     'triangle', 
#     'special_car',
#     'truck_box',
#     'motortricycle',
#     'tricycle',
#     'biking_person',
#     'car',
#     'truck_nobox_full',
#     'bus',
#     'ride_bicycle_person',
#     'walking_person',
#     'other']

CLASSES = [
    'cylinder',
    'triangle',
    'car',    # 小汽车
    'bus',    # 公交车
    'truck',  # 卡车
    'tricab', # 电动三轮
    'pedicab', # 人力三轮
    'other_vehicle', #其他车
    'pedistrian', # 行人
    'cyclist', #两轮车骑行人
]


CLASS_MAP = {
    'thick_cylinder':      'cylinder',
    'thin_cylinder':       'cylinder',
    'triangle':            'triangle',
    'special_car':         'car',
    'truck_box':           'truck',
    'motortricycle':       'tricab',
    'tricycle':            'pedicab',
    'biking_person':       'none',
    'car':                 'car',
    'truck_nobox_full':    'truck',
    'bus':                 'bus',
    'ride_bicycle_person': 'cyclist',
    'walking_person':      'pedistrian',
    'other':               'none'
    }



CATEGORIES = []
for idx, cls in enumerate(CLASSES):
    CATEGORIES.append(
        dict(
        id=idx+1,
        name=cls
        )
    )

def find_bl(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y):
    # find bottom left
    x_list = [pt1x, pt2x, pt3x, pt4x]
    y_list = [pt1y, pt2y, pt3y, pt4y]
    indexes = np.argsort(y_list)
    indexes = indexes[-2:].tolist()
    if x_list[indexes[0]]>x_list[indexes[1]]:
        return x_list[indexes[1]], y_list[indexes[1]]
    else:
        return x_list[indexes[0]], y_list[indexes[0]]

def find_br(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y):
    # find bottom right
    x_list = [pt1x, pt2x, pt3x, pt4x]
    y_list = [pt1y, pt2y, pt3y, pt4y]
    indexes = np.argsort(y_list)
    indexes = indexes[-2:].tolist()
    if x_list[indexes[0]]>x_list[indexes[1]]:
        return x_list[indexes[0]], y_list[indexes[0]]
    else:
        return x_list[indexes[1]], y_list[indexes[1]]

def viz(img_path, label_dict, out_str):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    rects = label_dict['ann']['bboxes']
    labels = label_dict['ann']['labels']
    anchors = label_dict['ann']['anchors']
    for rect, anchor in zip(rects, anchors):
        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[0]+rect[2]), int(rect[1]+rect[3])), (0, 255, 0), 2)
        cv2.circle(img, (anchor[0], anchor[1]), 3, (0,0,255), -1)

    cv2.imencode(img_path.suffix, img)[1].tofile(out_str)


def dumpAnnoMiddle(img_paths, outfile):
    items = []

    for idx in tqdm(range(len(img_paths))):
        img_path = img_paths[idx]
        height, width = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1).shape[:2]
        bboxes = []
        labels = []
        anchors = []
        label_file = img_path.with_suffix('.txt')
        lines = open(label_file, 'r').readlines()
        
        for line in lines:
            label = json.loads(line.strip())
            cls = label.get('shape', None) or label.get('nature', None)
            if cls is None:
                if 'vehicle_ped' in str(label_file.parent):
                    cls = 'car'
                else:
                    cls = 'triangle'
            if CLASS_MAP[cls] == 'none':
                continue
            cls_id = CLASSES.index(CLASS_MAP[cls])+1
            pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y = list(map(int,label['rect'].strip().split(',')))
            x_min = int(np.clip(min(pt1x, pt2x, pt3x, pt4x), 0, width-1))
            x_max = int(np.clip(max(pt1x, pt2x, pt3x, pt4x), 0, width-1))
            y_min = int(np.clip(min(pt1y, pt2y, pt3y, pt4y), 0, height-1))
            y_max = int(np.clip(max(pt1y, pt2y, pt3y, pt4y), 0, height-1))
            bboxes.append([x_min, y_min, x_max-x_min, y_max- y_min])
            labels.append(cls_id)
            if label.get('rect_in', None) is not None:
                if len(label['rect_in']) >1:
                    dist = -10
                    ptx_best = -1
                    pty_best = -1
                    for key in label['rect_in'].keys():
                        ptx, pty = list(map(int, label['rect_in'][key].strip().split(',')))
                        cur_dist = abs(ptx - x_min)
                        if cur_dist > dist:
                            dist = cur_dist
                            ptx_best = ptx
                            pty_best = pty
                    anchors.append([ptx_best, pty_best])
                else:
                    anchors.append(list(map(int, label['rect_in']['point0'].strip().split(','))))
            elif label.get('shape_type') == 'rect':
                x_c = (x_min + x_max)/2.0
                if x_c > width/2.0:
                    anchors.append([pt4x, pt4y])
                else:
                    anchors.append([pt3x, pt3y])
            else:
                x_c = (x_min + x_max)/2.0
                if x_min < width-x_max:
                    ptx, pty = find_bl(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y)
                else:
                    ptx, pty = find_br(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y)
                anchors.append([ptx, pty])

        bboxes = np.array(bboxes).astype(np.float32)
        labels = np.array(labels).astype(np.int64)
        anno_dict = dict(
            bboxes=bboxes,
            labels=labels,
            anchors=anchors
        )

        dummy_dict = dict(
            filename=str(img_paths[idx]),
            width=width,
            height=height,
            ann=anno_dict
        )
        items.append(dummy_dict)
    
        # viz(img_paths[idx], dummy_dict, 'test.png')
    mmcv.dump(items, outfile)
    # with open(outfile, 'w', encoding='utf8') as f:
    #     f.write(json.dumps(items, ensure_ascii=False))  


def cvtLabelTool2MiddleFormat(root):
    if isinstance(root, str):
        root = Path(root)

    train_outfile = root.joinpath('train_anno_middle.json')
    val_outfile = root.joinpath('val_anno_middle.json')
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
    dumpAnnoMiddle(img_paths[:num_train], train_outfile)
    dumpAnnoMiddle(img_paths[num_train:], val_outfile)

def get_args(arglist=None):
    parser = argparse.ArgumentParser('None')
    parser.add_argument('root')
    return parser.parse_args() if arglist is None else parser.parse_args(arglist)

if __name__ == '__main__':
    arglist = [
        '/mnt/104sdf/apaObjectDetection/haomo/trafficone'
        # '/mnt/104sdf/apaObjectDetection/haomo/vehicle_ped'
    ]
    args = get_args(arglist)
    cvtLabelTool2MiddleFormat(args.root)