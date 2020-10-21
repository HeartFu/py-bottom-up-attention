import base64
import csv
import json
import sys

from demo.relationship_classifier.VisualGenome import VisualGenome
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

import numpy as np
import torch
import cv2

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


if __name__ == '__main__':
    # vg = VisualGenome(relation_ann_file, relation_vocab_file, object_vocab_file)
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

    json_obj = json.load(open('/home/fanfu/newdisk/dataset/VisualGenome/objects.json', 'r'))
    # relation_ann_file = '/home/fanfu/newdisk/dataset/VisualGenome/relationships.json'
    # relation_vocab_file = '../../data/genome/1600-400-20/relations_vocab.txt'
    # object_vocab_file = '../../data/genome/1600-400-20/objects_vocab.txt'
    # vg = VisualGenome(relation_ann_file, relation_vocab_file, object_vocab_file)
    # relationships_all = vg.get_relationships_all()
    for item in json_obj:
        if item['image_id'] == 2417995 or item['image_id'] == 2417997 or item['image_id'] == 2417994:
            im = cv2.imread('/home/fanfu/newdisk/dataset/VisualGenome/VG_100K/' + str(item['image_id']) + '.jpg')
            if im is None:
                im = cv2.imread('/home/fanfu/newdisk/dataset/VisualGenome/VG_100K_2/' + str(item['image_id']) + '.jpg')

            objects = item['objects']
            for obj_item in objects:
                obj_id = obj_item['object_id']
                box = [[obj_item['x'], obj_item['y'], obj_item['x'] + obj_item['w'], obj_item['y'] + obj_item['h']]]
                # box = np.array([[1, 36, 481, 252]])
                feature = doit_boxes(im, predictor, box)

                fw = open('../data/test_boxes/'+ str(obj_id) + '.txt', 'w')
                fw.write(str(base64.b64encode(feature), encoding="utf-8"))
                fw.close()

    # # {"synsets": ["window.n.01"], "h": 50, "object_id": 3041406, "names": ["window"], "w": 16, "y": 100, "x": 314}