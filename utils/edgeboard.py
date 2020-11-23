#!/usr/bin/python
# -*- coding: UTF-8 -*- 

import cv2
import numpy as np

import paddlemobile as pm

predictor = None
labels = []


def init(configs):
    global predictor

    model_dir = configs['model']
    pm_config = pm.PaddleMobileConfig()
    pm_config.precision = pm.PaddleMobileConfig.Precision.FP32
    pm_config.device = pm.PaddleMobileConfig.Device.kFPGA
    pm_config.prog_file = model_dir + "/model"
    pm_config.param_file = model_dir + '/params'
    pm_config.thread_num = 4

    print('')
    print('configuration for predictor is :')
    print('\tPrecision: ' + str(pm_config.precision))
    print('\t   Device: ' + str(pm_config.device))
    print('\t    Model: ' + str(pm_config.prog_file))
    print('\t   Params: ' + str(pm_config.param_file))
    print('\tThreadNum: ' + str(pm_config.thread_num))
    print('')

    predictor = pm.CreatePaddlePredictor(pm_config)


def read_labels(configs):
    global labels
    if not 'labels' in configs:
        return
    label_path = configs['labels']
    if label_path is None or label_path == '':
        return
    with open(label_path) as label_file:
        line = label_file.readline()
        while line:
            labels.append(line.strip().split(':')[-1])
            line = label_file.readline()


def read_image(configs):
    print(configs)
    image_input = cv2.imread(configs['image'], cv2.IMREAD_COLOR)
    return image_input


def preprocess_image(image_input, configs):
    # resizing image
    print('image shape input: ' + str(image_input.shape))
    width = configs['input_width']
    height = configs['input_height']
    image_resized = cv2.resize(image_input, (width, height), cv2.INTER_CUBIC)
    print('image shape resized: ' + str(image_resized.shape))

    # to float32
    image = image_resized.astype('float32')

    # transpose to channel-first format
    image_transposed = np.transpose(image, (2, 0, 1))
    print('image shape transposed: ' + str(image_transposed.shape))

    # mean and scale preprocessing
    mean = np.array(configs['mean']).reshape((3, 1, 1))
    scale_number = configs['scale']
    scale = np.array([scale_number, scale_number, scale_number]).reshape((3, 1, 1))

    # RGB or BGR formatting
    format = configs['format'].upper()
    if format == 'RGB':
        b = image_transposed[0]
        g = image_transposed[1]
        r = image_transposed[2]
        image_transposed = np.stack([r, g, b])
        print('image shape formatted transposed: ' + str(image_transposed.shape))

    # mean and scale
    print('substract mean', mean.flatten(), ' and multiple with scale', scale.flatten())
    image_transposed -= mean
    image_transposed *= scale

    # transpose back
    image_result = np.transpose(image_transposed, (1, 2, 0))
    print('image shape transposed-back: ' + str(image_result.shape))

    return image_result


def draw_results(image, output, threshold):
    height, width, _ = image.shape
    print('boxes with scores above the threshold (%f): \n' % threshold)
    i = 1
    for box in output:
        if box[1] > threshold:
            print('\t', i, '\t', box[0], '\t', box[1], '\t', box[2], '\t', box[3], '\t', box[4], '\t', box[5])
            x_min = int(box[2] * width)
            y_min = int(box[3] * height)
            x_max = int(box[4] * width)
            y_max = int(box[5] * height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            i += 1
    cv2.imwrite('result.jpg', image)
    print('')


def classify(output, configs):
    data = output.flatten()
    max_index = 0
    score = 0.0
    for i in range(len(data)):
        if data[i] > score and not data[i] == float('inf'):
            max_index = i
            score = data[i]

    print('label: ', labels[max_index])
    print('index: ', max_index)
    print('score: ', score)
    print('')


def detect(output, configs):
    image = read_image(configs)
    draw_results(image, output, configs['threshold'])


def predict(configs, detection):
    global predictor
    width = configs['input_width']
    height = configs['input_height']

    image = read_image(configs)
    input = preprocess_image(image, configs)

    tensor = pm.PaddleTensor()
    tensor.dtype = pm.PaddleDType.FLOAT32
    tensor.shape = (1, 3, width, height)
    tensor.data = pm.PaddleBuf(input)

    paddle_data_feeds = [tensor]

    print('prediction is running ...')
    outputs = predictor.Run(paddle_data_feeds)
    assert len(outputs) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'

    output = np.array(outputs[0], copy=False)
    print('output:', output)
    print('\nprediction result :')
    print('\t nDim: ' + str(output.ndim))
    print('\tShape: ' + str(output.shape))
    print('\tDType: ' + str(output.dtype))

    # print(output)
    # print('')
    if detection:
        detect(output, configs)
    else:
        classify(output, configs)
