import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--img_path', type=str, default='/home/fanfu/newdisk/dataset/VisualGenome/',
                        help='image path')
    parser.add_argument('--split_path', type=str, default='data/',
                        help='')
    # parser.add_argument('--boxes_path', type=str, default='/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/boxes_feature/',
    #                     help='The path of bounding boxes feature')
    # parser.add_argument('--union_path', type=str, default='/import/nobackup_mmv_ioannisp/tx301/vg_feature/data/data/union_boxes/',
    #                     help='The path of union boxes feature')
    # parser.add_argument('--data_path', type=str,
    #                     default='',
    #                     help='')
    # parser.add_argument('--boxes_path', type=str,
    #                     default='/home/fanfu/newdisk/pytorch-bottom-up-attention/py-bottom-up-attention/demo/relationship_classifier/data/boxes_feature/',
    #                     help='The path of bounding boxes feature')
    # parser.add_argument('--union_path', type=str,
    #                     default='/home/fanfu/newdisk/pytorch-bottom-up-attention/py-bottom-up-attention/demo/relationship_classifier/data/union_boxes',
    #                     help='The path of union boxes feature')

    parser.add_argument('--ann_file', type=str, default='/home/fanfu/newdisk/dataset/VisualGenome/relationships.json',
                        help='The path of annotation relationship file.')
    parser.add_argument('--vocab_path', type=str, default='../data/genome/1600-400-20/',
                        help='The path of vocabulary')

    parser.add_argument('--train_proportion', type=float, default=0.8,
                        help='The path of union boxes feature')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial Learning Rate')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='minibatch size')
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--mGPUs', type=bool, default=False,
                        help='multi GPU run')

    # Log
    parser.add_argument('--output_dir', type=str, default='log/',
                        help='dir which save the log file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/',
                        help='Checkpoint path')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether recovery traing process')

    args = parser.parse_args()

    return args
