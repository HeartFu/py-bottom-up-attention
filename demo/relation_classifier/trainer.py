import csv
import os
import time

import cv2
import numpy as np

import torch
import wandb
from torch.optim import lr_scheduler
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
import sys

import torch.nn as nn

import opts, utils
from dataset_vg import BoxesDataset
from model.Classifier import Classifier

from demo.relation_classifier.VisualGenome import VisualGenome
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')

def collate_fn(data):  # 这里的data是一个list， list的元素是元组，构成为(self.data, self.label)
    # collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等
    # img_id, im, obj_boxes, sub_boxes, union_boxes, labels
    max_length = 0
    for i in range(len(data)):
        length = len(data[i][2])
        if length > max_length:
            max_length = length

    img_id_list = []
    obj_boxes_list = []
    sub_boxes_list = []
    union_boxes_list = []
    labels_list = []

    for i in range(len(data)):
        img_id_list.append(data[i][0])
        obj_boxes = data[i][1].numpy()
        sub_boxes = data[i][2].numpy()
        union_boxes = data[i][3].numpy()
        labels = data[i][4].numpy()
        if len(obj_boxes) != max_length:
            obj_boxes = padding_bboxes(obj_boxes, max_length)

            sub_boxes = padding_bboxes(sub_boxes, max_length)

            union_boxes = padding_bboxes(union_boxes, max_length)

            labels = padding_labels(labels, max_length)

        obj_boxes_list.append(obj_boxes)
        sub_boxes_list.append(sub_boxes)
        union_boxes_list.append(union_boxes)
        labels_list.append(labels)

    return (img_id_list, torch.tensor(obj_boxes_list), torch.tensor(sub_boxes_list), torch.tensor(union_boxes_list), torch.tensor(labels_list))

    # data.sort(key=lambda x: len(x[0][0]), reverse=False)  # 按照数据长度升序排序
    # data_list = []
    # label_list = []
    # min_len = len(data[0][0][0]) # 最短的数据长度
    # for batch in range(0, len(data)): #
    #     data_list.append(data[batch][0][:, :min_len])
    #     # 如果self.data和self.label是 100对1 的话
    #     # label_list.append(data[batch][1][:min_len])
    # data_tensor = torch.tensor(data_list, dtype=torch.float32)
    # label_tensor = torch.tensor(label_list, dtype=torch.float32)
    # data_copy = (data_tensor, label_tensor)
    # return data

def padding_labels(labels, length):
    labels_len = len(labels)
    padding_labels = []
    for i in range(length - labels_len):
        padding_labels.append(-1)
    return np.append(labels, np.asarray(padding_labels), axis=0)

def padding_bboxes(obj_boxes, length):
    boxes_len = len(obj_boxes)
    list_boxes = obj_boxes.tolist()
    for i in range(length - boxes_len):
        list_boxes.append([-1, -1, -1, -1])

    return np.asarray(list_boxes)

def train(cfgs):
    # dataset
    vg = VisualGenome(cfgs.ann_file, cfgs.vocab_path + 'relations_vocab.txt', cfgs.vocab_path + 'objects_vocab.txt')
    train_dataset = BoxesDataset(vg, cfgs.split_path, cfgs.img_path, split='train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=cfgs.batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)

    val_dataset = BoxesDataset(vg, cfgs.split_path, cfgs.img_path, split='val')
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=cfgs.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = Classifier(0.5)
    if wandb is not None:
        wandb.watch(model)
    if cfgs.resume:
        checkpoint = torch.load(cfgs.checkpoint + 'checkpoint_final.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        learning_rate = checkpoint['learning_rate']
        train_loss_epoch = checkpoint['train_loss_epoch']
        train_acc_epoch = checkpoint['train_acc_epoch']
        test_acc_epoch = checkpoint['test_acc_epoch']
    else:
        epoch = 0
        learning_rate = cfgs.learning_rate
        train_loss_epoch = []
        train_acc_epoch = []
        test_acc_epoch = []

    if cfgs.mGPUs:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    best_acc = -1.0

    cfg = get_cfg()
    cfg.merge_from_file("../../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"
    predictor = DefaultPredictor(cfg)
    csv.field_size_limit(sys.maxsize)

    # for epoch in range(cfgs.max_epochs):
    while epoch < cfgs.max_epochs:
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        count = 0
        accuracy_count = 0
        # start_time = time.time()
        # end_time = time.time()
        progress_bar = tqdm(train_loader, desc='|Train Epoch {}'.format(epoch), leave=False)
        for i, batch in enumerate(progress_bar):
            # end_time = time.time()
            # print('Done (t={:0.2f}s)'.format(end_time - start_time))
            count += 1
            img_id, obj_boxes, sub_boxes, union_boxes, labels = batch
            # print(img_id)

            obj_boxes, sub_boxes, union_boxes, labels = obj_boxes.cuda(), sub_boxes.cuda(), union_boxes.cuda(), labels.cuda()

            with torch.no_grad():
                obj_feature, sub_feature, union_feature = extract_feature(img_id, predictor, obj_boxes, sub_boxes,
                                                                          union_boxes, cfgs)

            outputs = model(obj_feature, sub_feature, union_feature)

            # outputs_reshape = torch.reshape(outputs, (outputs.size(0) * outputs.size(1), outputs.size(2)))
            labels_reshape = torch.reshape(labels, (labels.size(0) * labels.size(1),))
            labels_nopad = labels_reshape[labels_reshape[:] >= 0]
            # labels_reshape = torch.reshape(labels_nopad, (labels_nopad.size(0) * labels_nopad.size(1),))

            optimizer.zero_grad()
            loss = criterion(outputs, labels_nopad.long())
            loss.backward()
            optimizer.step()

            # pring statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accuracy = torch.sum(predicted == labels_nopad).item()
            train_acc += accuracy

            info_log = {
                'train_loss': '{:.3f}'.format(loss.item()),
                'train_accuracy': '{:.3f}'.format(accuracy / labels_nopad.size(0))
            }

            progress_bar.set_postfix(info_log, refresh=True)
            if wandb is not None:
                wandb.log(info_log)

            # start_time = time.time()
            accuracy_count += labels_nopad.size(0)
            # if count > 10:
            #     break

        loss_aveg = float(train_loss) / count
        acc_aveg = float(train_acc) / accuracy_count
        print('Train Epoch: {}, train_loss: {}, train_accuracy: {}.'.format(epoch, loss_aveg, acc_aveg))
        train_loss_epoch.append(loss_aveg)
        train_acc_epoch.append(acc_aveg)
        if wandb is not None:
            wandb.log({
                'train_loss_epoch': loss_aveg,
                'train_acc_epoch': acc_aveg
            })
        # scheduler.step()
        # caculate the test accuracy
        model.eval()
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                test_total = 0
                test_correct = 0
                process_bar_test = tqdm(val_loader, desc='|Test Epoch {}'.format(epoch), leave=False)
                for i, batch in enumerate(process_bar_test):

                    img_id, obj_boxes, sub_boxes, union_boxes, labels = batch
                    # print(img_id)

                    obj_boxes, sub_boxes, union_boxes, labels = obj_boxes.cuda(), sub_boxes.cuda(), union_boxes.cuda(), labels.cuda()

                    with torch.no_grad():
                        obj_feature, sub_feature, union_feature = extract_feature(img_id, predictor, obj_boxes,
                                                                                  sub_boxes,
                                                                                  union_boxes, cfgs)

                    outputs = model(obj_feature, sub_feature, union_feature)

                    labels_reshape = torch.reshape(labels, (labels.size(0) * labels.size(1),))
                    labels_nopad = labels_reshape[labels_reshape[:] >= 0]
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels_nopad.size(0)
                    correct = torch.sum(predicted == labels_nopad).item()
                    test_correct += correct
                    process_bar_test.set_postfix({'test_accuracy': '{:.3f}'.format(correct / labels_nopad.size(0))},
                                                 refresh=True)
                    # if count > 10:
                    #     break

                test_acc_aveg = float(test_correct) / test_total
                if wandb is not None:
                    wandb.log({
                        'test_acc_epoch': test_acc_aveg
                    })
                if acc_aveg > best_acc:
                    if cfgs.mGPUs:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'learning_rate': cfgs.learning_rate,
                            'loss': loss_aveg,
                            'accuracy': acc_aveg,
                            'test_accuracy': test_acc_aveg
                        }, cfgs.checkpoint + 'checkpoint_best.pkl')
                    else:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'learning_rate': cfgs.learning_rate,
                            'loss': loss_aveg,
                            'accuracy': acc_aveg,
                            'test_accuracy': test_acc_aveg
                        }, cfgs.checkpoint + 'checkpoint_best.pkl')
                print('Epoch: {}, Accuracy of the model on testset: {}'.format(epoch, test_acc_aveg))
                test_acc_epoch.append(test_acc_aveg)

        epoch += 1

    if epoch == cfgs.max_epochs:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'learning_rate': cfgs.learning_rate,
            'train_loss_epoch': train_loss_epoch,
            'train_acc_epoch': train_acc_epoch,
            'test_acc_epoch': test_acc_epoch
        }, cfgs.checkpoint + 'checkpoint_final.pkl')


# def remove_padding_label(labels):

def remove_padding_bbox(bboxes):
    # index = 0
    # for i in range(bboxes.size(0)):
    #     if bboxes[i] == [-1, -1, -1, -1]:
    #         index = i
    #         break
    return bboxes[bboxes[:, 0] >= 0]


def extract_feature(img_id, predictor, obj_boxes, sub_boxes, union_boxes, cfgs):
    batch = len(img_id)

    flag = 0 # set init feature tensor
    for i in range(batch):
        start_time = time.time()
        im = cv2.imread(os.path.join(cfgs.img_path, 'VG_100K', str(img_id[i]) + '.jpg'))
        if im is None:
            im = cv2.imread(os.path.join(cfgs.img_path, 'VG_100K_2', str(img_id[i]) + '.jpg'))
        if im is None:
            continue
        if flag == 0:
            obj_feature = doit_boxes(im, predictor, remove_padding_bbox(obj_boxes[i]))
            sub_feature = doit_boxes(im, predictor, remove_padding_bbox(sub_boxes[i]))
            union_feature = doit_boxes(im, predictor, remove_padding_bbox(union_boxes[i]))
            flag = 1
            # obj_feature = torch.unsqueeze(obj_feature, 0)
            # sub_feature = torch.unsqueeze(sub_feature, 0)
            # union_feature = torch.unsqueeze(union_feature, 0)
        else:
            obj_feature_item = doit_boxes(im, predictor, remove_padding_bbox(obj_boxes[i]))
            sub_feature_item = doit_boxes(im, predictor, remove_padding_bbox(sub_boxes[i]))
            union_feature_item = doit_boxes(im, predictor, remove_padding_bbox(union_boxes[i]))
            obj_feature = torch.cat((obj_feature, obj_feature_item), 0)
            sub_feature = torch.cat((sub_feature, sub_feature_item), 0)
            union_feature = torch.cat((union_feature, union_feature_item), 0)

        print('Done (t={:0.2f}s)'.format(time.time() - start_time))
    # obj_feature_reshape = torch.reshape(obj_feature, (-1, obj_feature.size(2)))
    # sub_feature_reshape = torch.reshape(sub_feature, (-1, sub_feature.size(2)))
    # union_feature_reshape = torch.reshape(union_feature, (-1, union_feature.size(2)))

    return obj_feature, sub_feature, union_feature


# def handle_boxes_img(raw_image, predictor, raw_boxes):
#     raw_boxes = Boxes(raw_boxes)
#     with torch.no_grad():
#         raw_height, raw_width = raw_image.shape[:2]
#         # Preprocessing
#         image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
#         new_height, new_width = image.shape[:2]
#         scale_x = 1. * new_width / raw_width
#         scale_y = 1. * new_height / raw_height
#         # print(scale_x, scale_y)
#         boxes = raw_boxes.clone()
#         boxes.scale(scale_x=scale_x, scale_y=scale_y)
#
#         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
#     return image, boxes

def doit_boxes(raw_image, predictor, raw_boxes):
    # raw_boxes = Boxes(torch.from_numpy(np.asarray(raw_boxes)).cuda())
    raw_boxes = Boxes(raw_boxes)
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

        return feature_pooled


if __name__ == '__main__':
    wandb.init(project="test")
    opt = opts.parse_opt()
    sys.stdout = utils.Logger(opt.output_dir + 'dropout/info.log', sys.stdout)
    sys.stderr = utils.Logger(opt.output_dir + 'dropout/error.log', sys.stderr)
    train(opt)
