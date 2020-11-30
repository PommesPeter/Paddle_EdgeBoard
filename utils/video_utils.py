#!/usr/bin/python
# -*- coding: UTF-8 -*-
import select
import threading

import cv2
import numpy as np
import v4l2capture
import warnings

from utils import print_results, draw_results
from video_serial import SerialThread

__all__ = ['video_process', 'VideoThread']


class VideoThread(threading.Thread):
    def __init__(self, video_device, video_w, video_h, buffer_size, name):
        threading.Thread.__init__(self)
        self.name = name
        self.is_loop = True
        self.daemon = True
        self.video = self.video_cap(video_device, video_w, video_h, buffer_size)
        self.frame = self.read_frame()
        print('初始化视频线程成功！')

    def run(self):
        # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        while self.is_loop:
            self.frame = self.read_frame()

    def stop(self):
        """
        结束线程
        """
        self.is_loop = False

    def video_cap(self, video_device, video_w, video_h, buffer_size):
        '''
        启动摄像头录像
        参数：设备路径
        返回：视频流
        '''
        video = v4l2capture.Video_device(video_device)
        video.set_format(video_w, video_h)
        video.create_buffers(buffer_size)
        video.queue_all_buffers()
        video.start()
        return video

    def get_image(self):
        return self.frame

    def read_frame(self):
        """
        this method is for reading frame
        :return: None
        """
        select.select((self.video,), (), ())
        image_data = self.video.read_and_queue()
        array = np.array(np.frombuffer(image_data, dtype=np.uint8))
        frame = array.reshape(960, 1280, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


def video_process(video_path, pm_model, save_video_flag):
    video_thread = VideoThread(video_path, 1280, 960, 1, '视频线程')
    video_thread.start()
    serial_thread = SerialThread('串口线程')
    serial_thread.start()
    init_flag = True

    while True:
        frame_read = video_thread.get_image()

        if frame_read is None:
            print('获取视频失败！')
            break

        # if init_flag and save_video_flag:
        #     # 视频模式输出检测视频
        #     save_name = 'save_video.avi'
        #     print('保存视频到' + save_name)
        #     out_video = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        #                                 (frame_read.shape[1], frame_read.shape[0]))
        #     init_flag = False
        if init_flag:
            init_flag = False
            continue
        # [类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标]
        boxes = pm_model.predict(frame_read)
        print_results(boxes, pm_model.label_names, init_flag)
        draw_results(frame_read, boxes, pm_model.colors, pm_model.label_names, False)
        serial_thread.set_data(boxes)

        # if save_video_flag:
        #     out_video.write(frame_read)


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # if not name.endswith('.avi'):  # 保证文件名的后缀是.avi
        #     name += '.avi'
        #     warnings.warn('video name should end with ".avi"')
        # elif not name.endswith('.mp4'):
        #     name += '.mp4'
        #     warnings.warn('video name should end with ".mp4"')
        self.__name = name  # 文件名
        self.__height = height  # 高
        self.__width = width  # 宽
        if name.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 如果是avi视频，编码需要为MJPG
        elif name.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频， 编码需要mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()
