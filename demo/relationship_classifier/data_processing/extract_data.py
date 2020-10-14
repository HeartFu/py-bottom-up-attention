"""
Extract Data: Object-Subject-union-boxes, id, relationship, imageId, objectId, features
"""
import base64
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
from demo.relationship_classifier.VisualGenome import VisualGenome
from detectron2.structures import Boxes
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

FIELDNAMES = ['image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'classes']

img_path = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K"
img_path2 = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K_2"
# img_path = "/home/scratch/VisualGenome/images/VG_100K"
# img_path2 = "/home/scratch/VisualGenome/images/VG_100K_2"
UNION_BOXES_PATH = '../data/union_boxes/'
BOXES_FEATURE_PATH = '../data/boxes_feature/'

THRESH = 0.4

NON_RELATION = 20


def get_boxes_data(infile, predictor, vg):
    # test_list = []
    # csv.field_size_limit(sys.maxsize)
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field]), dtype=np.float32).reshape(
                    (item['num_boxes'], -1))
            # in_data[item['image_id']] = item
            item['classes'] = np.frombuffer(base64.b64decode(item['classes']), dtype=np.int).reshape(
                (item['num_boxes'], -1))
            image_id = item['image_id']
            # test_list.append(item.copy())
            if os.path.exists(img_path + '/' + str(image_id) + '.jpg'):
                image_path = img_path + '/' + str(image_id) + '.jpg'
            else:
                image_path = img_path2 + '/' + str(image_id) + '.jpg'

            im = cv2.imread(image_path)
            if im is None:
                continue

            # vg.visualize(im, item['boxes'], item['classes'])

            get_union_boxes(item, im, predictor, vg)

            save_boxes_feature(item)


def save_boxes_feature(item):
    for i in range(item['num_boxes']):
        obj_id = str(item['image_id']) + '_' + str(i)
        save_boxes_feature_to_txt(item['features'][i], obj_id)


def save_boxes_feature_to_txt(feature, obj_id):
    fw = open(BOXES_FEATURE_PATH + obj_id + '.txt', 'w')
    fw.write(str(base64.b64encode(feature), encoding="utf-8"))
    fw.close()


"""
Extract union boxes information and feature
"""
def get_union_boxes(item, im, predictor, vg):
    len_boxes = item['num_boxes']
    # union_box_dict = dict()
    for i in range(len_boxes):
        obj_box = item['boxes'][i]
        for j in range(len_boxes):
            if i == j:
                continue

            union_id = str(item['image_id']) + '_' + str(i) + '_' + str(j)
            # object_id = item['image_id'] + '_' + str(i)
            # subject_id = item['image_id'] + '_' + str(j)

            sub_box = item['boxes'][j]
            x1 = obj_box[0].item() if obj_box[0] < sub_box[0] else sub_box[0].item()
            y1 = obj_box[1].item() if obj_box[1] < sub_box[1] else sub_box[1].item()
            x2 = obj_box[2].item() if obj_box[2] > sub_box[2] else sub_box[2].item()
            y2 = obj_box[3].item() if obj_box[3] > sub_box[3] else sub_box[3].item()

            # extract feature on ResNet
            union_feature = doit_union_boxes(im, predictor, [[x1, y1, x2, y2]])
            # save_union_feature(union_feature, union_id)

            # get relationship
            predicate = get_relationship(obj_box, sub_box, item['classes'][i][0], item['classes'][j][0], item['image_id'], vg)
            #
            # union_box_dict[union_id] = {
            #     'image_id': item['image_id'],
            #     'predicate': predicate,
            #     'feature': str(base64.b64encode(union_feature.numpy()), encoding="utf-8")
            # }

            # save the union_boxes information and feature
            json_obj = {
                'image_id': item['image_id'],
                'predicate': predicate,
                'feature': str(base64.b64encode(union_feature.numpy()), encoding="utf-8")
            }

            b = json.dumps(json_obj)
            f2 = open(UNION_BOXES_PATH + union_id + '.json', 'w')
            f2.write(b)
            f2.close()

#
# def save_boxes_feature(feature, obj_id):
#     fw = open(BOXES_FEATURE_PATH + obj_id + '.txt', 'w')
#     fw.write(str(base64.b64encode(feature), encoding="utf-8"))
#     fw.close()


def compute_iou(box1, box2):
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # computing the sum_area
    sum_area = s_box1 + s_box2

    # find the each edge of intersect rectangle
    left_line = max(box1[0], box2[0])
    right_line = min(box1[2], box2[2])
    top_line = max(box1[1], box2[1])
    bottom_line = min(box1[3], box2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def get_relationship(obj_box, sub_box, obj_class, sub_class, image_id, vg):
    relationship = vg.get_relationship(image_id)
    for relation in relationship:
        if relation['object']['name'] == obj_class and relation['subject']['name'] == sub_class:
            iou_obj = compute_iou(obj_box, relation['object']['boxes'])
            iou_sub = compute_iou(sub_box, relation['subject']['boxes'])
            if iou_obj > THRESH and iou_sub > THRESH:
                # same object
                return relation['predicate']

    return NON_RELATION


def doit_union_boxes(raw_image, predictor, raw_boxes):
    raw_boxes = Boxes(torch.from_numpy(np.asarray(raw_boxes)).cuda())

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])

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
        print('features:', features['res4'].shape)

        # Generate proposals with RPN
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)

        return feature_pooled.to('cpu')


def main(relation_ann_file, relation_vocab_file, object_vocab_file):
    vg = VisualGenome(relation_ann_file, relation_vocab_file, object_vocab_file)
    infile = '../../feature_VG_1.tsv'

    cfg = get_cfg()
    cfg.merge_from_file("../../../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
    predictor = DefaultPredictor(cfg)
    csv.field_size_limit(sys.maxsize)

    get_boxes_data(infile, predictor, vg)

if __name__ == '__main__':
    relation_ann_file = '/home/fanfu/newdisk/dataset/VisualGenome/relationships.json'
    relation_vocab_file = '../../data/genome/1600-400-20/relations_vocab.txt'
    object_vocab_file = '../../data/genome/1600-400-20/objects_vocab.txt'
    main(relation_ann_file, relation_vocab_file, object_vocab_file)
