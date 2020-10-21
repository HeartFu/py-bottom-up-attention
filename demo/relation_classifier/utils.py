import sys


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_union_box(obj_box, sub_box):
    x1 = obj_box[0] if obj_box[0] < sub_box[0] else sub_box[0]
    y1 = obj_box[1] if obj_box[1] < sub_box[1] else sub_box[1]
    x2 = obj_box[2] if obj_box[2] > sub_box[2] else sub_box[2]
    y2 = obj_box[3] if obj_box[3] > sub_box[3] else sub_box[3]

    # extract feature on ResNet
    # union_feature = doit_boxes(im, predictor, [[x1, y1, x2, y2]])

    return [x1, y1, x2, y2]