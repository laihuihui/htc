# 	Hybrid Task Cascade for Instance Segmentation

[English](./README.md) | 简体中文

  * [Hybrid Task Cascade for Instance Segmentation](#hybrid-task-cascade-for-instance-segmentation)
     * [一、简介](#一简介)
     * [二、复现精度](#二复现精度)
     * [三、数据集](#三数据集)
     * [四、环境依赖](#四环境依赖)
     * [五、快速开始](#五快速开始)
        * [step1: clone](#step1-clone)
        * [step2: 训练](#step2-训练)
        * [step3: 评估](#step3-评估)
     * [六、代码结构与详细说明](#六代码结构与详细说明)
        * [6.1 代码主要结构](#61-代码主要结构)
        * [6.2 部分参数说明](#62-部分参数说明)
        * [6.3 训练流程](#63-训练流程)
           * [单机训练](#单机训练)
           * [多机训练](#多机训练)
           * [训练输出](#训练输出)
        * [6.4 评估流程](#64-评估流程)
     * [七、模型信息](#七模型信息)
  

## 一、简介

本项目基于paddledetection框架复现HTC。HTC是一种目标检测实例分割网络，在 cascade rcnn 基础上修改 cascade head（加入mask预测部分，mask之间加入信息传递），并增加分支利用语义分割信息提供空间上下文信息。

**论文:**
- [1] K. Chen et al., “Hybrid Task Cascade for Instance Segmentation,” ArXiv190107518 Cs, Apr. 2019, Accessed: Aug. 31, 2021. [Online]. Available: http://arxiv.org/abs/1901.0751 <br>

**参考项目：**
- [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2253839](https://aistudio.baidu.com/aistudio/projectdetail/2253839)
- 脚本任务：[https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473)

## 二、复现精度

|  model   | Style  | box AP  | mask AP  |
|  ----  | ----  | ----  | ----  |
| htc-R-50-FPN(official)  | pytorch | 42.3 | 37.4 |
| **htc-R-50-FPN(mine)**  | Paddlepaddle | **42.6** | **37.9** |

**权重及日志下载**
权重地址：[百度网盘](https://pan.baidu.com/s/1RtCYvey8PXRbfgHJe4ujIQ) (提取码：yc1r )

权重对应：
```
weights
├── checkpoints
│   ├── htc_r50_fpn_1x_coco_resnet.pdparams (转换后的预训练权重参数)
│   ├── htc_r50_fpn_1x_coco.pdparams (转换后的官方权重参数)
├── output
│   ├── htc_r50_fpn_1x_coco
│   │   ├── model_final.pdparams (训练得到的权重参数)
```

## 三、数据集

[COCO 2017](https://cocodataset.org/#download) + [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)

- 数据集大小：
  - 训练集：118287张
  - 验证集：5000张
- 数据格式：图片

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.1.2

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/laihuihui/htc.git
cd htc
```
**安装依赖**
```bash
pip install -r requirements.txt
```

### step2: 训练
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml
```
如果训练中断通过 --resume 或 -r 参数恢复，例如使用下述命令在第3epoch中断则：
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml --resume output/htc_r50_fpn_1x_coco/3 
```
如果想要边训练边评估，可添加 --eval 参数实现：
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml --eval
```
如果你想分布式训练并使用多卡：
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml
```
此时的输出为：
```
Epoch: [0] [   0/7329] learning_rate: 0.000020 loss_rpn_cls: 0.691306 loss_rpn_reg: 0.054590 loss_bbox_cls_stage0: 4.189201 loss_bbox_reg_stage0: 0.064000 loss_bbox_cls_stage1: 2.481206 loss_bbox_reg_stage1: 0.016608 loss_bbox_cls_stage2: 1.106741 
```

### step3: 评估
```bash
python tools/eval.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o weights=output/htc_r50_fpn_1x_coco/model_final.pdparams
```

## 六、代码结构与详细说明

### 6.1 代码主要结构

```
├─config                          # 配置
├─dataset                         # 数据集加载
├─ppdet                           # 模型
├─output                          # 权重结果输出
├─log                             # 日志输出
├─tools                           # 工具代码
│   ├─eval.py                     # 评估
│   ├─train.py                    # 训练
│  README.md                      # 英文readme
│  README_cn.md                   # 中文readme
│  README_paddeldetection_cn.md   # pd英文readme
│  README_paddeldetection_cn.md   # pd中文readme
│  requirement.txt                # 依赖
```

### 6.2 部分参数说明

可以在 `tools/train.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 |
|  ----  |  ----  |  ----  |
| -c| None, 必选| 配置文件路径 |
| --eval| False, 可选 | 是否在训练时评估 |
| --resume 或 -r| None, 可选 | 恢复训练 |

### 6.3 训练流程

#### 单机训练
```bash
python tools/train.py -c $config_file
```

#### 多机训练
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c $config_file
```

此时，程序会将每个进程的输出log导入到`./log`路径下：
```
.
├── log
│   ├── endpoints.log
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
```

#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`batch`训练将会打印当前epoch、step以及loss值。
```text
Epoch: [0] [   0/7329] learning_rate: 0.000020 loss_rpn_cls: 0.691306 loss_rpn_reg: 0.054590 loss_bbox_cls_stage0: 4.189201 loss_bbox_reg_stage0: 0.064000 loss_bbox_cls_stage1: 2.481206 loss_bbox_reg_stage1: 0.016608 loss_bbox_cls_stage2: 1.106741 
```

### 6.4 评估流程

```bash
python tools/eval.py -c $config_file -o weights=$weight_file
```
可以使用 [百度网盘](https://pan.baidu.com/s/1RtCYvey8PXRbfgHJe4ujIQ) (提取码：yc1r ) 中的 `weights/output/htc_r50_fpn_1x_coco/model_final.pdparams` 预训练模型进行评估


## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | huihui lai|
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 目标检测、实例分割 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [百度网盘](https://pan.baidu.com/s/1RtCYvey8PXRbfgHJe4ujIQ) (提取码：yc1r )  |
| 在线运行 | [notebook任务](https://aistudio.baidu.com/aistudio/projectdetail/2253839) 、 [脚本任务](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473)|
