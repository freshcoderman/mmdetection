# -*- encoding: utf-8 -*-
'''
@File    :   getStatisticInfos.py
@Time    :   2023/03/02 14:16:13
@Author  :   Gang Xu 
@Version :   1.0
@Contact :   xg9712@arcsoft.com.cn
@Desc    :   None
'''

import os
import argparse
from pathlib import Path
import subprocess
import json
import sys
from collections import defaultdict as dfdict

class INFO:
    def __init__(self):
        self.root = ''
        self.num_images = 0
        self.num_classes = 0
        self.classes = set()
        self.num_objects = dfdict(int)

def getStatisticInfos(root):
    if isinstance(root, str):
        root = Path(root)
    data_info_path = root.joinpath('data.info')
    data_info_details_path = root.joinpath('data.details')
    fwdtls = open(data_info_details_path, 'w', encoding='utf8')
    fwtotal = open(data_info_path, 'w', encoding='utf8')
    
    folders = [x for x in root.rglob('*') if x.is_dir() and x.name=='remapped']
    total_info = INFO()
    total_info.root = str(root)
    for folder in folders:
        part_info = INFO()
        part_info.root = str(folder)
        label_files = [x for x in folder.iterdir() if x.suffix == '.txt']
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                label = json.loads(line.strip())
                cls = label.get('shape', None) or label.get('nature', None)
                if cls is None:
                    if 'vehicle_ped' in str(label_file.parent):
                        cls = 'car'
                    else:
                        cls = 'other'
                    print(f'Can not find valid class name, regarded as {cls}, please check file {label_file}.\n')
                    print(line)
                part_info.classes.add(cls)
                part_info.num_objects[cls] +=1
                total_info.classes.add(cls)
                total_info.num_objects[cls] +=1
            part_info.num_images+=1
            total_info.num_images+=1
        part_info.num_classes = len(part_info.classes)
        dump_dict = part_info.__dict__
        dump_dict['classes'] = list(dump_dict['classes'])
        fwdtls.write(json.dumps(dump_dict, indent=4, ensure_ascii=False))
        fwdtls.write('\n')
    total_info.num_classes = len(total_info.classes)
    dump_dict = total_info.__dict__
    dump_dict['classes'] = list(dump_dict['classes'])
    fwtotal.write(json.dumps(dump_dict, indent=4))
    fwtotal.write('\n')
    fwdtls.close()
    fwtotal.close()
    pass

def get_args(arglist=None):
    parser = argparse.ArgumentParser('None')
    parser.add_argument('root')
    return parser.parse_args() if arglist is None else parser.parse_args(arglist)

if __name__ == '__main__':
    arglist = [
        '/mnt/104sdf/apaObjectDetection/haomo'
    ]
    args = get_args(arglist)
    getStatisticInfos(args.root)
