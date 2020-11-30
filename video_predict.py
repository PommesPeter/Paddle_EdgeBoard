#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import json
import sys

from utils.tools import *
from utils.utils import load_label_list, print_args


def parse_args():
    parser = argparse.ArgumentParser(description='API implementation for Paddle-Mobile')
    parser.add_argument('-j', '--json',
                        help='configuration file for the prediction', default="config/detection/vgg-ssd/screw.json")
    return parser.parse_args()


if __name__ == '__main__':
    label_map = dict()
    args = parse_args()
    print_args(args)
    if args.json is None or args.json == '':
        print('\nFor usage, please use the -h switch.\n\n')
        sys.exit(0)

    with open(args.json) as json_file:
        configs = json.load(json_file)
    label_map = load_label_list()
    if type(label_map) is not dict:
        raise TypeError('label_list is not correct')
    else:
        # print(label_map)
        pass
    baidu = Baidu(configs, label_map, 'video')
    baidu.read_labels(configs)
    baidu.predict_video(configs)
    sys.exit(0)
