# Image Segmentation Benchmark
## Prerequisites
Install all needed packages/libraries
```bash
pip install -r requirements.txt
```

## Dataset Used:
#### PASCAL Visual Object Classes
* Total number of classes: 20
* The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations.

## Models To Use:
* *fcn_resnet50* 
* *fcn_resnet101*
* *deeplabv3_resnet50*
* *deeplabv3_resnet101*
* *deeplabv3_mobilenet_v3_large*
* *lraspp_mobilenet_v3_large*
           

## How to Run:
```bash
python3 bench_PASCALVOC.py --model [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large] \
                           --pretrained [true, false]
```

## Results:
- *All models have been pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.*

| Model                        | Mean IoU | Global Accuracy | Class Accuracies                                                                                                                                                 | Class IoUs                                                                                                                                                       |
| ---------------------------- |:--------:|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| FCN-ResNet50                 | 72.8     | 00.0            |                                                                                                                                                                  |                                                                                                                                                                  |
| FCN-ResNet101                | 72.8     |   94.0          | ['97.2', '89.9', '86.4', '86.4', '82.1', '55.2', '91.7', '74.1', '92.0', '63.1', '91.2', '75.4', '86.4', '87.8', '83.3', '95.5', '65.7', '82.8', '72.7', '86.8'] | ['93.7', '88.3', '39.9', '82.4', '72.0', '52.0', '90.0', '68.6', '87.7', '42.7', '79.1', '61.5', '78.8', '77.7', '78.7', '89.4', '55.9', '79.4', '53.2', '84.0'] |
| Deeplabv3-ResNet50           | 75.6     |   94.4.         | ['96.3', '92.1', '88.2', '95.5', '83.8', '62.4', '98.8', '74.3', '94.7', '67.2', '91.5', '70.4', '92.3', '92.1', '95.0', '95.3', '74.2', '93.9', '79.4', '93.3']                | ['93.7', '90.0', '41.2', '87.1', '70.0', '56.8', '94.3', '65.5', '92.1', '46.8', '86.8', '53.5', '85.9', '81.0', '87.6', '88.8', '63.2', '88.5', '54.0', '85.6'] | 
| Deeplabv3-ResNet101          | 77.8     |   95.0          | -----:|
| DeepLabV3 MobileNet V3 Large | are neat |    $1           | -----:|
| LRASPP MobileNet V3 Large    | are neat |    $1           | -----:|


## License ![GitHub](https://img.shields.io/github/license/rainarit/segmentation-benchmark)

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/rainarit/segmentation-benchmark/blob/main/LICENSE) file for details
