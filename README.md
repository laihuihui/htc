# 	基于 PaddleDetection 的 Hybrid Task Cascade for Instance Segmentation 实现

---
## reference
- 论文：[Hybrid Task Cascade for Instance Segmentation](https://arxiv.org/abs/1901.07518)
- 官方源码：https://github.com/open-mmlab/mmdetection
---
## result
- 训练环境：V100*4（baidu ai studio 脚本）
- 训练结果对比：

|  model   | Style  | box AP  | mask AP  |
|  ----  | ----  | ----  | ----  |
| htc-R-50-FPN(official)  | pytorch | 42.3 | 37.4 |
| **htc-R-50-FPN(mine)**  | Paddlepaddle | **42.6** | **37.9** |

---
## download
- 使用数据集：[COCO 2017](https://cocodataset.org/#download) + [stuffthingmaps_trainval2017](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
    - 数据集文件路径
        ```angular2html
        htc
        ├── dataset
        │   ├── coco
        │   │   ├── annotations
        │   │   ├── train2017
        │   │   ├── val2017
        │   │   ├── stuffthingmaps
        │   │   │   ├── train2017
        │   │   │   ├── val2017
        ```
- 权重及 log：[百度网盘](https://pan.baidu.com/s/1-Szst_ODOlNFtTkTBGdfWw) (提取码：laih )
    - 权重文件路径：
        ```angular2html
        htc
        ├── checkpoints
        │   ├── htc_r50_fpn_1x_coco_resnet.pdparams (pretrain weight)
        ├── output
        │   ├── htc_r50_fpn_1x_coco
        │   │   ├── model_final.pdparams (final weight)
        ```
---

## train & eval
```
pip install -r requirements.txt
# 训练
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o --eval
# 验证
python tools/eval.py -c configs/htc/htc_r50_fpn_1x_coco.yml -o weights=output/htc_r50_fpn_1x_coco/model_final.pdparams
```