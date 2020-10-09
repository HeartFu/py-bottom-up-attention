import os
import io
import h5py

# detectron2 faster_rcnn interface
from detectron2.structures import Boxes
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch

# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def doit(raw_image, predictor):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        print('features:', features['res4'].shape)
        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]

        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        #         print(boxes)

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        #         for nms_thresh in np.arange(0.5, 1.0, 0.1):
        #             print(nms_thresh)
        instances, ids = fast_rcnn_inference_single_image(
            boxes, probs, image.shape[1:],
            score_thresh=0.2, nms_thresh=0.3, topk_per_image=-1
        )
        # print(len(ids))
        #         if len(ids) > 100:
        #             break
        #             if len(ids) == NUM_OBJECTS:
        #                 break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        print(instances)

        # union_bounding_boxes
        bboxes = instances.pred_boxes.tensor
        union_boxes = []
        union_boxes_index = []
        for i in range(len(bboxes)):
            obj_box = bboxes[i]
            for j in range(len(bboxes)):
                sub_box = bboxes[j]
                # compare the different object and subject to get the union box
                x1 = obj_box[0].item() if obj_box[0] < sub_box[0] else sub_box[0].item()
                y1 = obj_box[1].item() if obj_box[1] < sub_box[1] else sub_box[1].item()
                x2 = obj_box[2].item() if obj_box[2] > sub_box[2] else sub_box[2].item()
                y2 = obj_box[3].item() if obj_box[3] > sub_box[3] else sub_box[3].item()
                union_boxes.append([x1, y1, x2, y2])
                union_boxes_index.append((i, j))

        union_boxes_tensor = torch.from_numpy(np.array(union_boxes)).cuda()

        union_box_features = predictor.model.roi_heads._shared_roi_transform(
            features, [Boxes(union_boxes_tensor)]
        )
        union_box_features = union_box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Union Feature Size:', union_box_features.shape)

        return instances, roi_features, union_boxes_tensor, union_box_features, union_boxes_index


def list_data(predictor, path, h5file):
    files = os.listdir(path)
    i = 0
    for file in files:
        if not os.path.isdir(file):
            im = cv2.imread(path + '/' + file)
            instances, roi_features, union_boxes, union_boxes_feature, union_boxes_index = doit(im, predictor)
            image_id = file.replace('.jpg', '')
            data = {
                'image_id': image_id,
                'image_h': instances.image_height,
                'image_w': instances.image_width,
                'num_boxes': instances.num_instances,
                'boxes': instances.pred_boxes.tensor,
                'features': roi_features,
                'union_boxes': union_boxes,
                'union_boxes_feature': union_boxes_feature,
                'union_boxes_index': union_boxes_index
            }

            h5file.create_dataset(image_id, data=data)

            if i % 10000 == 0:
                print('Extracting Features for i.....')


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

    # extract features

    path = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K"
    path2 = "/home/fanfu/newdisk/dataset/VisualGenome/VG_100K_2"

    with h5py.File('feature_VG.h5', 'a') as h:
        print('Start Extract Feature for VG_100K')
        list_data(predictor, path, h)
        print('Start Extract Feature for VG_100K_2')
        list_data(predictor, path2, h)
