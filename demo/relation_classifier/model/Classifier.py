import csv
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes


class Classifier(nn.Module):

    def __init__(self, dropout):
        super(Classifier, self).__init__()

        cfg = get_cfg()
        cfg.merge_from_file("../../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        # VG Weight
        cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
        self.predictor = DefaultPredictor(cfg)
        csv.field_size_limit(sys.maxsize)

        self.fc_obj = nn.Linear(2048, 512)
        self.obj_drop = torch.nn.Dropout(dropout)
        self.fc_sub = nn.Linear(2048, 512)
        self.sub_drop = torch.nn.Dropout(dropout)
        self.fc_union = nn.Linear(2048, 512)
        self.union_drop = torch.nn.Dropout(dropout)

        self.fc_classification = nn.Linear(512 * 3, 21)
        self.dropout = dropout
        self._init_weights()

    def forward(self, im, obj_boxes, sub_boxes, union_boxes):

        # extract features
        # import pdb
        # pdb.set_trace()
        batch = im.shape[0]

        for i in range(batch):
            if i == 0:
                obj_feature = self.doit_boxes(im[i], obj_boxes[i])
                sub_feature = self.doit_boxes(im[i], sub_boxes[i])
                union_feature = self.doit_boxes(im[i], sub_boxes[i])
                obj_feature = torch.unsqueeze(obj_feature, 0)
                sub_feature = torch.unsqueeze(sub_feature, 0)
                union_feature = torch.unsqueeze(union_feature, 0)
            else:
                obj_feature_item = self.doit_boxes(im[i], obj_boxes[i])
                sub_feature_item = self.doit_boxes(im[i], sub_boxes[i])
                union_feature_item = self.doit_boxes(im[i], sub_boxes[i])
                obj_feature = torch.cat((obj_feature, torch.unsqueeze(obj_feature_item, 0)), 0)
                sub_feature = torch.cat((sub_feature, torch.unsqueeze(sub_feature_item, 0)), 0)
                union_feature = torch.cat((union_feature, torch.unsqueeze(union_feature_item, 0)), 0)

        obj_feature = F.relu(self.obj_drop(self.fc_obj(obj_feature)))
        sub_feature = F.relu(self.sub_drop(self.fc_sub(sub_feature)))
        union_feature = F.relu(self.union_drop(self.fc_union(union_feature)))

        feature = torch.cat((obj_feature, sub_feature, union_feature), 2)

        x = self.fc_classification(feature)

        return x

    def doit_boxes(self, raw_image, raw_boxes):
        # raw_boxes = Boxes(torch.from_numpy(np.asarray(raw_boxes)).cuda())
        raw_boxes = Boxes(raw_boxes)
        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            # print("Original image size: ", (raw_height, raw_width))

            # Preprocessing
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
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
            images = self.predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = self.predictor.model.backbone(images.tensor)
            # print('features:', features['res4'].shape)

            # Generate proposals with RPN
            proposal_boxes = [boxes]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            # print('Pooled features size:', feature_pooled.shape)

            return feature_pooled

    # initialize weights
    def _init_weights(self):
        for _m in self.modules():
            print(_m)
            if isinstance(_m, nn.Linear):
                nn.init.xavier_uniform_(_m.weight.data)
                _m.bias.data.fill_(0.1)
