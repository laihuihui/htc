# 	Hybrid Task Cascade for Instance Segmentation

English | [简体中文](./README_cn.md)

  * [Hybrid Task Cascade for Instance Segmentation](#hybrid-task-cascade-for-instance-segmentation)
     * [1 Introduction](#1-introduction)
     * [2 Accuracy](#2-accuracy)
     * [3 Dataset](#3-dataset)
     * [4 Environment](#4-environment)
     * [5 Quick start](#5-quick-start)
        * [step1: clone](#step1-clone)
        * [step2: train](#step2-train)
        * [step3: evaluation](#step3-evaluation)
     * [6 Code structure](#6-code-structure)
        * [6.1 main structure](#61-main-structure)
        * [6.2 Part of the parameter description](#62-part-of-the-parameter-description)
        * [6.3 Training process](#63-training-process)
           * [Single machine training](#single-machine-training)
           * [Multi machine training](#multi-machine-training)
           * [Training output](#training-output)
        * [6.4 assessment process](#64-assessment-process)
     * [7 Model information](#7-model-information)
  

## 1 Introduction
This project reproduces HTC based on paddledetection framework. 

Cascade is a classic yet powerful architecture that has
boosted performance on various tasks. However, how to introduce
cascade to instance segmentation remains an open
question. A simple combination of Cascade R-CNN and
Mask R-CNN only brings limited gain. In
this work, authors propose a new framework, Hybrid Task Cascade
(HTC), which differs in two important aspects: (1) instead
of performing cascaded refinement on these two tasks
separately, it interweaves them for a joint multi-stage processing;
(2) it adopts a fully convolutional branch to provide
spatial context, which can help distinguishing hard
foreground from cluttered background. Overall, this framework
can learn more discriminative features progressively
while integrating complementary features together in each
stage. Without bells and whistles, a single HTC obtains
38.4% and 1.5% improvement over a strong Cascade Mask
R-CNN baseline on MSCOCO dataset. Moreover, the overall
system achieves 48.6 mask AP on the test-challenge split,
ranking 1st in the COCO 2018 Challenge Object Detection
Task.

**Paper:**
- [1] K. Chen et al., “Hybrid Task Cascade for Instance Segmentation,” ArXiv190107518 Cs, Apr. 2019, Accessed: Aug. 31, 2021. [Online]. Available: http://arxiv.org/abs/1901.0751 <br>

**Reference project:**
- [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

**The link of aistudio:**
- notebook: [https://aistudio.baidu.com/aistudio/projectdetail/2253839](https://aistudio.baidu.com/aistudio/projectdetail/2253839)
- Script: [https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473)

## 2 Accuracy

|  model   | Style  | box AP  | mask AP  |
|  ----  | ----  | ----  | ----  |
| htc-R-50-FPN(official)  | pytorch | 42.3 | 37.4 |
| **htc-R-50-FPN(mine)**  | Paddlepaddle | **42.6** | **37.9** |

**Model & Log Download Address**
[Baidu Web Drive](https://pan.baidu.com/s/1fThnatGEWrfFm3Q1fagBjQ) (access code: yc1r )

Detailed information：
```
weights
├── checkpoints
│   ├── htc_r50_fpn_1x_coco_resnet.pdparams
│   ├── htc_r50_fpn_1x_coco.pdparams
├── output
│   ├── htc_r50_fpn_1x_coco
│   │   ├── model_final.pdparams
```

## 3 Dataset

[COCO 2017](https://cocodataset.org/#download) + [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)

- Dataset size：
  - train: 118287
  - val: 5000
- Data format：picture

## 4 Environment

- Hardware: GPU, CPU

- Framework:
  - PaddlePaddle >= 2.1.2

## 5 Quick start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/laihuihui/htc.git
cd htc
```
**Installation dependency**
```bash
pip install -r requirements.txt
```

### step2: train
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml
```
If the training is interrupted, it can be recovered through the `-- resume` parameter or `-r` parameter, for example, using the `-- resume output/htc_r50_fpn_1x_coco/3` means the interrupt is resumed at 3epoch:
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml --resume output/htc_r50_fpn_1x_coco/3 
```
Perform evaluation in train using `--eval` parameter:
```bash
python tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml --eval
```
If you want to train distributed and use multicards:
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml
```
The output is:
```
Epoch: [0] [   0/7329] learning_rate: 0.000020 loss_rpn_cls: 0.691306 loss_rpn_reg: 0.054590 loss_bbox_cls_stage0: 4.189201 loss_bbox_reg_stage0: 0.064000 loss_bbox_cls_stage1: 2.481206 loss_bbox_reg_stage1: 0.016608 loss_bbox_cls_stage2: 1.106741 
```

### step3: evaluation
```bash
python tools/eval.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o weights=output/htc_r50_fpn_1x_coco/model_final.pdparams
```

## 6 Code structure

### 6.1 main structure

```
├─config                          
├─dataset                         
├─ppdet                           
├─output                          
├─log                             
├─tools                           
│   ├─eval.py                     
│   ├─train.py                    
│  README.md                      
│  README_cn.md                   
│  README_paddeldetection_cn.md   
│  README_paddeldetection_cn.md   
│  requirement.txt                
```

### 6.2 Part of the parameter description

Parameters related to training and evaluation can be set in `tools/train.py`, as follows:

|  Parameters   | default  | description |
|  ----  |  ----  |  ----  |
| -c| None, Mandatory| Configuration file path |
| --eval| False, Optional | Whether to perform evaluation in train |
| --resume or -r| None, Optional | Recovery training |

### 6.3 Training process

#### Single machine training
```bash
python tools/train.py -c $config_file
```

#### Multi machine training
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c $config_file
```

At this time, the program will import the output log of each process into the path of `./log`:
```
.
├── log
│   ├── endpoints.log
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   └── workerlog.3
```

#### Training output
After the training starts, you will get the output similar to the following. Each round of 'batch' training will print the current epoch, step and loss values.
```text
Epoch: [0] [   0/7329] learning_rate: 0.000020 loss_rpn_cls: 0.691306 loss_rpn_reg: 0.054590 loss_bbox_cls_stage0: 4.189201 loss_bbox_reg_stage0: 0.064000 loss_bbox_cls_stage1: 2.481206 loss_bbox_reg_stage1: 0.016608 loss_bbox_cls_stage2: 1.106741 
```

### 6.4 assessment process

```bash
python tools/eval.py -c $config_file -o weights=$weight_file
```
Pre training model: `weights/output/htc_r50_fpn_1x_coco/model_final.pdparams` in [Baidu Web Drive](https://pan.baidu.com/s/1fThnatGEWrfFm3Q1fagBjQ) (access code: yc1r )

## 7 Model information

For other information about the model, please refer to the following table:

| information | description |
| --- | --- |
| Author | huihui lai|
| Date | 2021.08 |
| Framework version | Paddle 2.1.2 |
| Application scenarios | Object detection , Instance Segmentation |
| Support hardware | GPU, CPU |
| Download link | [Pre training model & Logs](https://pan.baidu.com/s/1fThnatGEWrfFm3Q1fagBjQ) (access code: yc1r )  |
| Online operation | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/2253839) , [Script](https://aistudio.baidu.com/aistudio/clusterprojectdetail/2270473)|
