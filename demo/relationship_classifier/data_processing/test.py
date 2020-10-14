# import os
#
# from detectron2.data import MetadataCatalog
#
# vg_classes = []
# with open('../../data/genome/1600-400-20/objects_vocab.txt') as f:
#     for object in f.readlines():
#         vg_classes.append(object.split(',')[0].lower().strip())
#
# MetadataCatalog.get("vg").thing_classes = vg_classes
# print(vg_classes)
from tqdm import tqdm

d = {'k1':1, 'k2':2}
for k, v in tqdm(d.items()):
    print(k, v)