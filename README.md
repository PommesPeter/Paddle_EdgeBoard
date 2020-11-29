# Paddle_EdgeBoard

> This repo is for the usage of EdgeBoard which is from baidu

## Project Structure

> 用于EdgeBoard上的代码结构，该项目已上传Github：！[link](https://github.com/PommesPeter/Paddle_EdgeBoard)

```
Paddle_EdgeBoard
├───config  (存储所有的配置文件)
│   └───detection   (存储目标检测的配置文件)
│       ├───mobilenet-ssd
│       ├───mobilenet-ssd-640
│       ├───ssd_mobilenet_v1_voc    (我们的模型所用的配置文件)
│       └───vgg-ssd
├───data    (存放要识别的图片或者视频)
├───model   (存放模型文件)
│   └───detection
│       ├───mobilenet-ssd
│       ├───mobilenet-ssd-640
│       ├───ssd_mobilenet_v1_voc    (我们使用的模型)
│       └───vgg-ssd
├───output  (输出识别之后的结果或视频)
├───utils   (实用工具代码)
├───image_predict.py    (检测单张图片)
├───video_predict.py    (检测视频)
├───camera_predict.py   (摄像头检测)
└───README.md   (说明文档)
```