import json
import time

import matplotlib.pyplot as plt

class VisualGenome:
    def __init__(self, annotation_file=None, relations_vocab_file=None, object_vocab_file=None):
        self.dataset, self.relation_map_class, self.object_map_class = dict(), dict(), dict()
        # read the relations_vocab
        if not relations_vocab_file is None:
            file = open(relations_vocab_file, 'r')
            lines = file.readlines()
            for i in range(len(lines)):
                line = lines[i]
                relations = line.split(',')
                for relation in relations:
                    self.relation_map_class[relation.lower().strip()] = i
            file.close()

        if not object_vocab_file is None:
            file = open(object_vocab_file, 'r')
            lines = file.readlines()
            for i in range(len(lines)):
                line = lines[i]
                objects = line.split(',')
                for obj in objects:
                    self.object_map_class[obj.lower().strip()] = i
            file.close()

        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            # assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.creat_index(dataset)

    def creat_index(self, dataset):
        count_classes = [0 for _ in range(21)]
        for item in dataset:
            relation_list = []
            for relationship in item['relationships']:
                obj = relationship['object']
                sub = relationship['subject']
                if relationship['predicate'].lower().strip() in self.relation_map_class.keys():
                    predicte = self.relation_map_class[relationship['predicate'].lower().strip()]
                else:
                    predicte = 20
                count_classes[predicte] += 1

                if obj['name'].lower().strip() not in self.object_map_class.keys() or sub['name'].lower().strip() not in self.object_map_class:
                    continue

                relation = {
                    'predicate': predicte,
                    'object': {
                        'name': self.object_map_class[obj['name'].lower().strip()],
                        'boxes': [obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']],
                        'object_id': obj['object_id']
                    },
                    'subject': {
                        'name': self.object_map_class[sub['name'].lower().strip()],
                        'boxes': [sub['x'], sub['y'], sub['x'] + sub['w'], sub['y'] + sub['h']],
                        'object_id': sub['object_id']
                    },
                    'relationship_id': relationship['relationship_id']
                }
                relation_list.append(relation)
            self.dataset[item['image_id']] = relation_list
        print(count_classes)

    def get_relationship(self, image_id):
        return self.dataset[image_id]

    def get_relationships_all(self):
        return self.dataset

    def get_relationship_vocab(self):
        relation_class = []
        for relation in self.relation_map_class.values():
            if relation not in relation_class:
                relation_class.append(relation)

        return relation_class

    def bbox_to_rect(self, bbox, color):  # 本函数已保存在d2lzh包中方便以后使用
        # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
        # ((左上x, 左上y), 宽, 高)
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
            fill=False, edgecolor=color, linewidth=2)

    def visualize(self, im, boxes, classes):
        fig = plt.imshow(im)
        for i, box in enumerate(boxes):
            rect = self.bbox_to_rect(box, 'red')
            fig.axes.add_patch(rect)
            fig.axes.text(rect.xy[0] + 24, rect.xy[1] + 10, classes[i][0],
                          va='center', ha='center', fontsize=6, color='blue',
                          bbox=dict(facecolor='m', lw=0))
        plt.show()