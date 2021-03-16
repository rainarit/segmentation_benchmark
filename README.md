# Image Segmentation Benchmark
## Installation
Install all needed packages/libraries
```bash
pip install -r requirements.txt
```

## Dataset Used:
#### PASCAL Visual Object Classes
* ##### Total number of classes: 20
* ##### The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations.

## Models To Use:
* #### *fcn_resnet50* 
* #### *fcn_resnet101*
* #### *deeplabv3_resnet50*
* #### *deeplabv3_resnet101*
* #### *deeplabv3_mobilenet_v3_large*
* #### *lraspp_mobilenet_v3_large*
           

## How to Run:
```bash
python3 torchbench_PASCALVOC.py --model [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large]
```

