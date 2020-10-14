import base64
import csv
import os
import io
import sys


# detectron2 faster_rcnn interface
from detectron2.structures import Boxes

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch


def doit_union_boxes(raw_image, predictor, raw_boxes):
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())

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


def get_union_boxes(bboxes):
    union_boxes = []
    union_boxes_index = []
    for i in range(len(bboxes)):
        obj_box = bboxes[i]
        for j in range(len(bboxes)):
            if i == j:
                continue
            sub_box = bboxes[j]
            # compare the different object and subject to get the union box
            x1 = obj_box[0].item() if obj_box[0] < sub_box[0] else sub_box[0].item()
            y1 = obj_box[1].item() if obj_box[1] < sub_box[1] else sub_box[1].item()
            x2 = obj_box[2].item() if obj_box[2] > sub_box[2] else sub_box[2].item()
            y2 = obj_box[3].item() if obj_box[3] > sub_box[3] else sub_box[3].item()
            union_boxes.append([x1, y1, x2, y2])
            union_boxes_index.append((i, j))

    return np.asarray(union_boxes), np.asarray(union_boxes_index)


if __name__ == '__main__':
    # Load VG Classes
    data_path = 'data/genome/1600-400-20'

    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())

    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes
    MetadataCatalog.get("vg").attr_classes = vg_attrs

    cfg = get_cfg()
    cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
    predictor = DefaultPredictor(cfg)
    csv.field_size_limit(sys.maxsize)

    FIELDNAMES = ['image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'classes']
    FIELDNAMES_UNION_BOXES = ['image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'union_index']
    infile = 'feature_VG.tsv'

    # img_path = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K"
    # img_path2 = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K_2"
    img_path = "/home/scratch/VisualGenome/images/VG_100K"
    img_path2 = "/home/scratch/VisualGenome/images/VG_100K_2"

    # in_data = {}
    i = 0
    with open('feature_union_boxes_VG.tsv', 'w') as tsv_write_file:
        writer = csv.DictWriter(tsv_write_file, delimiter='\t', fieldnames=FIELDNAMES_UNION_BOXES)
        with open(infile, "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.b64decode(item[field]), dtype=np.float32).reshape((item['num_boxes'], -1))
                # in_data[item['image_id']] = item
                item['classes'] = np.frombuffer(base64.b64decode(item['classes']), dtype=np.int).reshape((item['num_boxes'], -1))
                image_id = item['image_id']
                if os.path.exists(img_path + '/' + str(image_id) + '.jpg'):
                    image_path = img_path + '/' + str(image_id) + '.jpg'
                else:
                    image_path = img_path2 + '/' + str(image_id) + '.jpg'

                im = cv2.imread(image_path)
                if im is None:
                    continue
                union_boxes, union_boxes_index = get_union_boxes(item['boxes'])

                index = 0
                while index < len(union_boxes):
                    if index + 1000 > len(union_boxes):
                        last_index = len(union_boxes)
                    else:
                        last_index = index + 1000

                    features = doit_union_boxes(im, predictor, union_boxes[index:last_index])
                    if index == 0:
                        roi_features = features.clone().numpy()
                    else:
                        roi_features = np.concatenate((roi_features, features.clone().numpy()), axis=0)

                    index += 1000
                # union_boxes, union_boxes_index, roi_features = doit_union_boxes(im, predictor, item['boxes'])
                print('start save data!')
                data = {
                    'image_id': image_id,
                    'image_h': item['image_h'],
                    'image_w': item['image_w'],
                    'num_boxes': len(union_boxes),
                    'boxes': str(base64.b64encode(union_boxes), encoding='utf-8'),
                    'features': str(base64.b64encode(roi_features), encoding='utf-8'),
                    'union_index': str(base64.b64encode(union_boxes_index), encoding='utf-8')
                    # 'union_boxes': base64.b64encode(union_boxes.to('cpu').numpy()),
                    # 'union_boxes_feature': base64.b64encode(union_boxes_feature.to('cpu').numpy()),
                    # 'union_boxes_index': base64.b64encode(np.array(union_boxes_index))
                }

                writer.writerow(data)

                if i % 10000 == 0:
                    print('Extracting Features for i.....{}', i)

                i += 1

