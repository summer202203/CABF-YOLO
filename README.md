# 运行程序
```
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd /content/drive/MyDrive/YOLOX
pip3 install -v -e .
python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py --fp16 -o
```
# 数据集
* NEU-DET数据集官网地址：[NEU-DET使用说明](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
* 数据集采用VOC形式，整理后的数据集可直接用于代码：[谷歌云盘](https://drive.google.com/drive/folders/1PCIGSFXW0SkgDWaUckYHXluPAwK6oqfY?usp=sharing)

# 可视化
## 使用tensorboard
```
Load the TensorBoard notebook extension
load_ext tensorboard
tensorboard --logdir=YOLOX_outputs/yolox_ca/tensorboard/ --port='6001' --bind_all
```
# 消融实验
## 注意力机制消融实验
1、\yolox\models\yolo_pafpn.py
```
        # 直接对输入的特征图使用注意力机制
        x0 = self.cam_1(x0)
        # x0 = self.cbam_1(x0)
        # x0 = self.se_1(x0)
        # x0 = self.eca_1(x0)
        #################################
```
2、yolox\models\network_blocks.py：不同注意力机制的具体算法代码模块
## TCCA、BF、EIOU消融实验
TCCA: \yolox\models\yolo_pafpn.py
```
        # 直接对输入的特征图使用注意力机制
        x0 = self.cam_1(x0)
```
BF: \yolox\models\yolo_pafpn.py
```
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        pan_out0 = self.C3_In4(p_out0)  # 1024->1024/32
```
EIOU: yolox\models\losses.py
```
    def __init__(self, reduction="none", loss_type="eiou"):
```

# 最佳实验结果
```
Writing crazing VOC results file
Writing inclusion VOC results file
Writing patches VOC results file
Writing pitted_surface VOC results file
Writing rolled-in_scale VOC results file
Writing scratches VOC results file
Eval IoU : 0.50
AP for crazing = 0.6082
AP for inclusion = 0.9546
AP for patches = 0.9801
AP for pitted_surface = 0.9935
AP for rolled-in_scale = 0.8130
AP for scratches = 0.9913
Mean AP = 0.8901
~~~~~~~~
Results:
0.608
0.955
0.980
0.994
0.813
0.991
0.890
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
Recompute with `./tools/reval.py --matlab ...` for your paper.
-- Thanks, The Management
--------------------------------------------------------------
Eval IoU : 0.55
Eval IoU : 0.60
Eval IoU : 0.65
Eval IoU : 0.70
Eval IoU : 0.75
Eval IoU : 0.80
Eval IoU : 0.85
Eval IoU : 0.90
Eval IoU : 0.95
--------------------------------------------------------------
map_5095: 0.5424730153762637
map_50: 0.890143085180931
--------------------------------------------------------------
2023-08-07 19:25:00 | INFO     | yolox.core.trainer:354 - 
Average forward time: 4.84 ms, Average NMS time: 0.86 ms, Average inference time: 5.70 ms

2023-08-07 19:25:00 | INFO     | yolox.core.trainer:364 - Save weights to ./YOLOX_outputs/yolox_voc_s
2023-08-07 19:25:03 | INFO     | yolox.core.trainer:195 - Training of experiment is done and the best AP is 54.25
```
