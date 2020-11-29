#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import cv2
import numpy as np
import paddlemobile as pm

# from utils.video_utils import VideoThread, SerialThread
from video_utils import VideoThread, SerialThread


class Baidu:
    def __init__(self, configs, label_map, name):

        self.name = name
        self.predictor = None
        self.labels = []
        self.classes = []
        self.video_path = configs['video']
        self.configs = configs
        self.model_dir = configs['model']
        self.label_map = label_map
        self.pm_config = pm.PaddleMobileConfig()
        self.pm_config.precision = pm.PaddleMobileConfig.Precision.FP32
        self.pm_config.device = pm.PaddleMobileConfig.Device.kFPGA
        self.pm_config.prog_file = os.path.join(self.model_dir, 'model')
        self.pm_config.param_file = os.path.join(self.model_dir, 'params')
        self.pm_config.thread_num = 4

        print('configuration for predictor is :')
        print('\tPrecision: ' + str(self.pm_config.precision))
        print('\t   Device: ' + str(self.pm_config.device))
        print('\t    Model: ' + str(self.pm_config.prog_file))
        print('\t   Params: ' + str(self.pm_config.param_file))
        print('\tThreadNum: ' + str(self.pm_config.thread_num))

        self.predictor = pm.CreatePaddlePredictor(self.pm_config)
        self.tensor = self.init_tensor((1, 3, configs['input_width'], configs['input_height']))
        # init video_thread
        if self.name == 'video':
            self.video_thread = VideoThread(self.video_path, self.configs['input_width'], self.configs['input_height'], 1,
                                        'video_thread')

    def init_tensor(self, data_shape):
        tensor = pm.PaddleTensor()
        tensor.dtype = pm.PaddleDType.FLOAT32
        tensor.shape = data_shape
        return tensor

    def read_labels(self, configs):
        if not 'labels' in configs:
            return
        label_path = configs['labels']
        if label_path is None or label_path == '':
            return
        with open(label_path) as label_file:
            line = label_file.readline()
            while line:
                self.labels.append(line.strip().split(':')[-1])
                line = label_file.readline()

    def read_image(self, configs):
        image_input = cv2.imread(configs['image'], cv2.IMREAD_COLOR)
        return image_input

    def preprocess_image(self, image_input, configs):
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
        # print('substract mean', mean.flatten(), ' and multiple with scale', scale.flatten())
        image_transposed -= mean
        image_transposed *= scale

        # transpose back
        image_result = np.transpose(image_transposed, (1, 2, 0))
        print('image shape transposed-back: ' + str(image_result.shape))

        return image_result

    def draw_results(self, image, output, threshold):

        height, width, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('boxes with scores above the threshold (%f): ' % threshold)
        i = 1
        print('类别\t置信度\t中点坐标\t左上坐标\t右下坐标\t')
        for box in output:
            if box[1] > threshold:
                print(self.label_map[str(int(box[0]))], '\t', box[1], '\t', box[2], '\t', box[3], '\t', box[4], '\t', box[5])
                x_min = int(box[2] * width)
                y_min = int(box[3] * height)
                x_max = int(box[4] * width)
                y_max = int(box[5] * height)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(image, self.label_map[str(int(box[0]))] + ":" + "{:.2f}".format(box[1]), (x_min, y_min - 10), font, 1,
                            (0, 255, 0), 6)
                i += 1
        cv2.imwrite("/home/root/workspace/Paddle_EdgeBoard/output/result.jpg", image)

    def show_result_in_console(self, image, output, threshold):
        height, width, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('[INFO] boxes with scores above the threshold (%f): ' % threshold)
        print('类别\t置信度\t中点坐标\t左上坐标\t右下坐标\t')
        for box in output:
            if box[1] > threshold:
                print(self.label_map[str(int(box[0]))], '\t', box[1], '\t', box[2], '\t', box[3], '\t', box[4], '\t', box[5])

    def show_result_in_video(self, image, output, threshold):
        height, width, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        print('boxes with scores above the threshold (%f): ' % threshold)
        i = 1
        print('类别\t置信度\t中点坐标\t左上坐标\t右下坐标\t')
        for box in output:
            if box[1] > threshold:
                x_min = int(box[2] * width)
                y_min = int(box[3] * height)
                x_max = int(box[4] * width)
                y_max = int(box[5] * height)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(image, self.label_map[str(int(box[0]))] + ":" + "{:.2f}".format(box[1]), (x_min, y_min - 10), font, 1,
                            (0, 255, 0), 6)

    def detect(self, output, configs):
        image = self.read_image(configs)
        self.draw_results(image, output, configs['threshold'])

    def predict_image(self, configs):

        image = self.read_image(configs)
        input = self.preprocess_image(image, configs)

        self.tensor.data = pm.PaddleBuf(input)

        paddle_data_feeds = [self.tensor]

        print('prediction is running ...')
        outputs = self.predictor.Run(paddle_data_feeds)
        assert len(outputs) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'

        output = np.array(outputs[0], copy=False)

        print('\nprediction result :')
        print('\t nDim: ' + str(output.ndim))
        print('\tShape: ' + str(output.shape))
        print('\tDType: ' + str(output.dtype))

        self.detect(output, configs)

    def predict_video(self, configs):

        self.video_thread.start()
        init_flag = True
        while True:
            frame = self.video_thread.get_image()
            if frame is None:
                print('[INFO] fail to read frame ...')
                break
            if init_flag:
                print('[INFO] prediction is running ...')
                init_flag = False
                continue

            image = self.preprocess_image(frame, configs)
            self.tensor.data = pm.PaddleBuf(image)
            paddle_data_feeds = [self.tensor]

            outputs = self.predictor.Run(paddle_data_feeds)

            assert len(outputs) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'

            output = np.array(outputs[0], copy=False)
            self.show_result_in_console(frame, output, configs['threshold'])

    def predict_camera(self):
        camera_serial = SerialThread('serial_thread')
        camera_serial.start()
        self.video_thread.start()
        pass
