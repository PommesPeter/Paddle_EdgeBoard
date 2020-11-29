# Paddle_EdgeBoard

> This repo is for the usage of EdgeBoard which is from baidu

## Project Structure

> 用于EdgeBoard上的代码结构，该项目已上传Github：！[link](https://github.com/PommesPeter/Paddle_EdgeBoard)

```
Paddle_EdgeBoard
├───config  
│   └───detection   (存储目标检测的配置文件)
│       ├───mobilenet-ssd
│       ├───mobilenet-ssd-640
│       ├───ssd_mobilenet_v1_voc    (我们的模型所用的配置文件)
│       └───vgg-ssd
├───data
├───model
│   └───detection
│       ├───mobilenet-ssd
│       ├───mobilenet-ssd-640
│       ├───ssd_mobilenet_v1_voc    (我们使用的模型)
│       └───vgg-ssd
├───test
├───output  
├───utils
├───image_predict.py
├───video_predict.py
├───camera_predict.py
└───README.md
```

- config: 存储所有的配置文件
- data: 存放要识别的图片或者视频
- model: 存放模型文件
- test: 用于存放单元测试代码
- output: 输出识别之后的结果或视频
- utils: 实用工具代码
- image_predict.py: 检测单张图片
- video_predict.py: 检测视频
- camera_predict.py: 摄像头检测
- README.md: 说明文档


## Code Introduction

为了能够实使用EdgeBoard我们翻阅了百度AI Studio中与EdgeBoard相关的文档，由于EdgeBoard的文档在AI Studio中设计得不太好友好，需要一直翻遍所有网页才找到了所有相关的文档。所以我们从查阅资料到设计花费了很多时间。为了能够跑起来我们的代码需要把EdgeBoard的环境配置好，配置EdgeBoard的环境需要做以下几个工作：

- 配置网络环境
- 装载FPGA驱动
- 配置Python环境


> 配置过程中可能也会出现其他的问题。
