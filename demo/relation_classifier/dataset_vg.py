import json
import os
import time

import cv2
import torch
import base64
import numpy as np

from torch.utils import data

from demo.relation_classifier.utils import get_union_box


class BoxesDataset(data.Dataset):

    def __init__(self, vg, split_file, image_path, split='train'):
        self.vg = vg
        relationships = vg.get_relationships_all()
        self.dataset = []
        data_file = open(os.path.join(split_file, split + '.txt'))
        data_lines = data_file.readlines()
        for data_line in data_lines:
            data_line_split = data_line.split(' ')
            img_path = data_line_split[0].split('/')
            img_name = int(img_path[1].replace('.jpg', ''))
            if img_name in relationships.keys():
                # relationships[img_name]['image_id'] = img_name
                self.dataset.append({
                    'image_id': img_name,
                    'relationships': relationships[img_name]
                })

        self.img_path = image_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_info = self.dataset[index]
        img_id = data_info['image_id']
        relations = data_info['relationships']
        obj_boxes = []
        sub_boxes = []
        union_boxes = []
        labels = []

        # for test
        obj_name = []
        sub_name = []
        union_boxes_name = []

        for item in relations:
            # get union_boxes
            union_box = get_union_box(item['object']['boxes'], item['subject']['boxes'])
            union_boxes.append(union_box)
            obj_boxes.append(item['object']['boxes'])
            sub_boxes.append(item['subject']['boxes'])
            labels.append(item['predicate'])

            # for test
            obj_name.append(item['object']['name'])
            sub_name.append(item['subject']['name'])
            union_boxes_name.append(item['predicate'])
        #
        # im = cv2.imread(os.path.join(self.img_path, 'VG_100K', str(img_id) + '.jpg'))
        # if im is None:
        #     im = cv2.imread(os.path.join(self.img_path, 'VG_100K_2', str(img_id) + '.jpg'))

        # extract features

        # self.vg.visualize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), obj_boxes, obj_name, 'obj_boxes.jpg')
        # self.vg.visualize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), sub_boxes, sub_name, 'sub_boxes.jpg')
        # self.vg.visualize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), union_boxes, union_boxes_name, 'union_boxes.jpg')

        return img_id, torch.from_numpy(np.array(obj_boxes)), torch.from_numpy(np.array(sub_boxes)), torch.from_numpy(np.array(union_boxes)), torch.from_numpy(np.array(labels))
