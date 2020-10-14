import base64
import csv
import json
import os
import sys
import cv2
import torch

import numpy as np
from tqdm import tqdm

from demo.relationship_classifier.VisualGenome import VisualGenome
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

img_path = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K"
img_path2 = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K_2"
# img_path = "/home/scratch/VisualGenome/images/VG_100K"
# img_path2 = "/home/scratch/VisualGenome/images/VG_100K_2"
UNION_BOXES_PATH = '../data/union_boxes/'
BOXES_FEATURE_PATH = '../data/boxes_feature/'


def extract_union_feature(vg, predictor):
    object_list = []
    union_boxes_info_all = []

    union_boxes_ids = [] # 用于存储feed进predictor的union boxes ID
    union_boxes = []
    boxes_ids = [] # 用于存储feed进predictor的boxes ID
    boxes_list = []

    relationships_all = vg.get_relationships_all()
    for img_id, relationships in tqdm(relationships_all.items()):
        # for img_id in relationships_all.keys():
        #     relationships = relationships_all[img_id]
        if os.path.exists(img_path + '/' + str(img_id) + '.jpg'):
            image_path = img_path + '/' + str(img_id) + '.jpg'
        else:
            image_path = img_path2 + '/' + str(img_id) + '.jpg'

        im = cv2.imread(image_path)
        if im is None:
            continue
        for relationship in relationships:
            obj_info = relationship['object']
            sub_info = relationship['subject']
            predicate = relationship['predicate']
            relation_id = relationship['relationship_id']

            union_box = get_union_box(obj_info['boxes'], sub_info['boxes'])
            union_boxes.append(union_box)
            union_boxes_ids.append(relation_id)
            if len(union_boxes) >= 50:
                union_boxes_feature = doit_boxes(im, predictor, union_boxes)
                for i in range(len(union_boxes)):
                    save_boxes_feature_to_txt(union_boxes_feature[i], union_boxes_ids[i], UNION_BOXES_PATH)
                union_boxes.clear()
                union_boxes_ids.clear()

            union_info = {
                'image_id': img_id,
                'object': obj_info['object_id'],
                'subject': sub_info['object_id'],
                'predicate': predicate,
                'union_boxes_id': relation_id
                # 'union_feature': str(base64.b64encode(union_boxes_feature.numpy()), encoding="utf-8")
            }

            union_boxes_info_all.append(union_info)

            # operate for boxes
            if obj_info['object_id'] not in object_list:
                boxes_ids.append(obj_info['object_id'])
                boxes_list.append(obj_info['boxes'])
                # obj_feature = doit_boxes(im, predictor, [obj_info['boxes']])
                # save_boxes_feature_to_txt(obj_feature, obj_info['object_id'], BOXES_FEATURE_PATH)
                object_list.append(obj_info['object_id'])

            if sub_info['object_id'] not in object_list:
                boxes_ids.append(sub_info['object_id'])
                boxes_list.append(sub_info['boxes'])
                # sub_feature = doit_boxes(im, predictor, [sub_info['boxes']])
                # save_boxes_feature_to_txt(sub_feature, sub_info['object_id'], BOXES_FEATURE_PATH)
                object_list.append(sub_info['object_id'])

            if len(boxes_ids) >= 50:
                boxes_feature = doit_boxes(im, predictor, boxes_list)
                for i in range(len(boxes_ids)):
                    save_boxes_feature_to_txt(boxes_feature[i], boxes_ids[i], BOXES_FEATURE_PATH)
                boxes_list.clear()
                boxes_ids.clear()

    # save all boxes
    b = json.dumps(union_boxes_info_all)
    f2 = open(UNION_BOXES_PATH + 'union_boxes_info.json', 'w')
    f2.write(b)
    f2.close()


def save_boxes_feature_to_txt(feature, obj_id, path):
    fw = open(path + str(obj_id) + '.txt', 'w')
    fw.write(str(base64.b64encode(feature), encoding="utf-8"))
    fw.close()


def get_union_box(obj_box, sub_box):
    x1 = obj_box[0] if obj_box[0] < sub_box[0] else sub_box[0]
    y1 = obj_box[1] if obj_box[1] < sub_box[1] else sub_box[1]
    x2 = obj_box[2] if obj_box[2] > sub_box[2] else sub_box[2]
    y2 = obj_box[3] if obj_box[3] > sub_box[3] else sub_box[3]

    # extract feature on ResNet
    # union_feature = doit_boxes(im, predictor, [[x1, y1, x2, y2]])

    return [x1, y1, x2, y2]


def doit_boxes(raw_image, predictor, raw_boxes):
    raw_boxes = Boxes(torch.from_numpy(np.asarray(raw_boxes)).cuda())

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])

        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        # print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # ---
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        # print('features:', features['res4'].shape)

        # Generate proposals with RPN
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        return feature_pooled.to('cpu').numpy()


def main(relation_ann_file, relation_vocab_file, object_vocab_file):
    vg = VisualGenome(relation_ann_file, relation_vocab_file, object_vocab_file)
    # infile = '../../feature_VG_1.tsv'

    cfg = get_cfg()
    cfg.merge_from_file("../../../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
    predictor = DefaultPredictor(cfg)
    csv.field_size_limit(sys.maxsize)

    extract_union_feature(vg, predictor)


if __name__ == '__main__':
    relation_ann_file = '/home/fanfu/newdisk/dataset/VisualGenome/relationships.json'
    relation_vocab_file = '../../data/genome/1600-400-20/relations_vocab.txt'
    object_vocab_file = '../../data/genome/1600-400-20/objects_vocab.txt'
    main(relation_ann_file, relation_vocab_file, object_vocab_file)
