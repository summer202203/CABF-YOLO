# Start to run the code
The code is based on the [YOLOX model](https://github.com/Megvii-BaseDetection/YOLOX), which has been improved and optimized for the strip steel surface defect dataset. Follow the steps below to download and run the code.
```
git clone https://github.com/summer202203/CABF-YOLO.git
cd CABF-YOLO
pip3 install -v -e .
python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py --fp16 -o
```
# Dataset
## In order to validate the robustness and generalization of the model, we validate the model on two publicly available strip steel surface defect datasets.
* NEU-DET dataset link: [NEU-DET](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
* GC10-DET dataset link: [GC10-DET](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets.git))
* The code can be in the form of both VOC and COCO data, and we used the VOC dataset form in our experiments, which can be used directly in the code after organizing the NEU-DET dataset: [Google Drive](https://drive.google.com/drive/folders/1PCIGSFXW0SkgDWaUckYHXluPAwK6oqfY?usp=sharing and GC10-DET dataset: [Google Drive](https://drive.google.com/drive/folders/1RkPZWtg4HK_Quq0t41WqS9-9GyYNFbId?usp=drive_link).

# Visualization
## Using tensorboard: you first need to download tensorboard, and then run the following command to realize the visualization operation.
```
# Load the TensorBoard notebook extension
tensorboard --logdir=YOLOX_outputs/tensorboard/ --port='6001' --bind_all
```
# Comparison experiment

## Comparison with SOTA object detectors.
Experiments are performed on the NEU-DET and GC10-DET datasets using the generic model, and the YOLOv3 to YOLOv8 models are available for download on your own.

##  Comparison of the effects of different attention mechanism modules.
To conduct experiments comparing attention mechanisms, only some of the code needs to be modified in the \yolox\models\yolo_pafpn.py. Add: the specific algorithms for the different attention mechanisms are implemented in yolox\models\network_blocks.py.
```
        # Use the attention mechanism directly on the input feature map
        x0 = self.cam_1(x0)
        # x0 = self.cbam_1(x0)
        # x0 = self.se_1(x0)
        # x0 = self.eca_1(x0)
        #################################
```

# Ablation experiment
### Experiments on verifying TCCA module, BF strategy and EIoU loss function. When performing ablation experiments, the code corresponding to each module is directly eliminated.
TCCA module: \yolox\models\yolo_pafpn.py
```
        # Use the attention mechanism directly on the input feature map
        x0 = self.cam_1(x0)
```
BF srategy: \yolox\models\yolo_pafpn.py
```
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        pan_out0 = self.C3_In4(p_out0)  # 1024->1024/32
```
EIOU loss function: yolox\models\losses.py
```
    def __init__(self, reduction="none", loss_type="eiou"):
```
