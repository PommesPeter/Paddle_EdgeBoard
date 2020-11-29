#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import json
import sys

from utils.tools import *
from utils.utils import load_label_list


def parse_args():
    parser = argparse.ArgumentParser(description='API implementation for Paddle-Mobile')
    parser.add_argument('-d', '--detection',
                        help='flag indicating detections',
                        default=True)
    parser.add_argument('-j', '--json',
                        help='configuration file for the prediction', default="./config/detection/ssd_moblienet_v1_voc/underwater.json")
    return parser.parse_args()


def print_args(args):
    print('Arguments: ')
    print('\t', '    detection flag: ', args.detection)
    print('\t', 'json configuration: ', args.json)


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    if args.json is None or args.json == '':
        print('\nFor usage, please use the -h switch.\n\n')
        sys.exit(0)

    label_map = load_label_list()
    if type(label_map) is not dict:
        raise TypeError('label_list is not correct')
        sys.exit(0)
    else:
        print(label_map)

    with open(args.json) as json_file:
        configs = json.load(json_file)
    baidu = Baidu(configs, label_map, 'image')
    baidu.read_labels(configs)
    baidu.predict_image(configs)
    sys.exit(0)
