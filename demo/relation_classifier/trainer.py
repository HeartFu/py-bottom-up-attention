import os
import time

import torch
from torch.optim import lr_scheduler
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
import sys

import opts, utils
from dataset_vg import BoxesDataset
from model.Classifier import Classifier

from demo.relation_classifier.VisualGenome import VisualGenome

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')


def train(cfgs):
    # dataset
    vg = VisualGenome(cfgs.ann_file, cfgs.vocab_path + 'relations_vocab.txt', cfgs.vocab_path + 'objects_vocab.txt')
    train_dataset = BoxesDataset(vg, cfgs.split_path, cfgs.img_path, split='train')
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, shuffle=False)

    val_dataset = BoxesDataset(vg, cfgs.split_path, cfgs.img_path, split='val')
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # Model
    model = Classifier(0.4)
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

    model.to(device)

    # optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=1e-5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    best_acc = -1.0

    # for epoch in range(cfgs.max_epochs):
    while epoch < cfgs.max_epochs:
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        count = 0
        # start_time = time.time()
        # end_time = time.time()
        progress_bar = tqdm(train_loader, desc='|Train Epoch {}'.format(epoch), leave=False)
        for i, batch in enumerate(progress_bar):
            # end_time = time.time()
            # print('Done (t={:0.2f}s)'.format(end_time - start_time))
            count += 1
            img_id, im, obj_boxes, sub_boxes, union_boxes, labels = batch

            im = im.numpy()
            obj_boxes, sub_boxes, union_boxes, labels = obj_boxes.to(device), sub_boxes.to(device), union_boxes.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(im, obj_boxes, sub_boxes, union_boxes)

            outputs_reshape = torch.reshape(outputs, (outputs.size(0) * outputs.size(1), outputs.size(2)))
            labels_reshape = torch.reshape(labels, (labels.size(0) * labels.size(1),))

            loss = criterion(outputs_reshape, labels_reshape)
            loss.backward()
            optimizer.step()

            # pring statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs_reshape, 1)
            accuracy = torch.sum(predicted == labels_reshape).item()
            train_acc += accuracy
            progress_bar.set_postfix({'train_loss': '{:.3f}'.format(loss.item()),
                                      'train_accuracy': '{:.3f}'.format(accuracy / float(cfgs.batch_size))
                                      },
                                     refresh=True)

            # start_time = time.time()
            break

        loss_aveg = float(train_loss) / count
        acc_aveg = float(train_acc) / (count * cfgs.batch_size)
        print('Train Epoch: {}, train_loss: {}, train_accuracy: {}.'.format(epoch, loss_aveg, acc_aveg))
        train_loss_epoch.append(loss_aveg)
        train_acc_epoch.append(acc_aveg)

        # scheduler.step()
        # caculate the test accuracy
        model.eval()
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                test_total = 0
                test_correct = 0
                process_bar_test = tqdm(val_loader, desc='|Test Epoch {}'.format(epoch), leave=False)
                for i, batch in enumerate(process_bar_test):
                    img_id, im, obj_boxes, sub_boxes, union_boxes, labels = batch
                    im = im.numpy()
                    obj_boxes, sub_boxes, union_boxes, labels = obj_boxes.to(device), sub_boxes.to(
                        device), union_boxes.to(device), labels.to(device)

                    outputs = model(im, obj_boxes, sub_boxes, union_boxes)

                    outputs_reshape = torch.reshape(outputs, (outputs.size(0) * outputs.size(1), outputs.size(2)))
                    labels_reshape = torch.reshape(labels, (labels.size(0) * labels.size(1),))
                    _, predicted = torch.max(outputs_reshape, 1)
                    test_total += labels.size(0)
                    correct = torch.sum(predicted == labels_reshape).item()
                    test_correct += correct
                    process_bar_test.set_postfix({'test_accuracy': '{:.3f}'.format(correct / float(cfgs.batch_size))},
                                                 refresh=True)
                    break

                test_acc_aveg = float(test_correct) / test_total
                if acc_aveg > best_acc:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'learning_rate': cfgs.learning_rate,
                        'loss': loss_aveg,
                        'accuracy': acc_aveg,
                        'loss_val': test_acc_aveg
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


if __name__ == '__main__':
    opt = opts.parse_opt()
    sys.stdout = utils.Logger(opt.output_dir + 'dropout/info.log', sys.stdout)
    sys.stderr = utils.Logger(opt.output_dir + 'dropout/error.log', sys.stderr)
    train(opt)
