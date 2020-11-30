#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import argparse
import json
import sys

import tools


def parse_args():
    parser = argparse.ArgumentParser(description='API implementation for Paddle-Mobile')
    parser.add_argument('-d', '--detection',
                        help='flag indicating detections',
                        action="store_true")
    parser.add_argument('-j', '--json',
                        help='configuration file for the prediction')
    return parser.parse_args()


def print_args(args):
    print('Arguments: ')
    print('\t', '    detection flag: ', args.detection)
    print('\t', 'json configuration: ', args.json)


def main():
    args = parse_args()
    print_args(args)
    if args.json is None or args.json == '':
        print('\nFor usage, please use the -h switch.\n\n')
        sys.exit(0)

    with open(args.json) as json_file:
        configs = json.load(json_file)

    tools.init(configs)
    tools.read_labels(configs)
    tools.predict(configs, args.detection)


if __name__ == '__main__':
    sys.exit(main())
