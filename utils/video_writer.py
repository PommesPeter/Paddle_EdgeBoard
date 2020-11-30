from __future__ import absolute_import, division, print_function

import warnings

import cv2
import numpy as np

warnings.simplefilter("always")


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        self.__name = name  # file name
        self.__height = height  # height
        self.__width = width  # width
        if name.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # encoing MJPG if your video is avi format
        elif name.endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # encoing mp4v if your video is mp4 format
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        """
        this method is using for write frame to a new video
        :param frame: the input of frame
        :return: None
        """
        # check for the frame type
        if frame.dtype != np.uint8:
            raise ValueError('frame.dtype should be np.uint8')
        # check for the shape of frame
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        """
        close video writer
        :return: None
        """
        self.__writer.release()
