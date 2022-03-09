# !/usr/bin/python

import os, sys

os.chdir("/home/AD/rraina/segmentation_benchmark/semseg/output")

for i in range(1, 21):
    contrast1 = i

    if contrast1 == 1:
        contrast2 = 1
    else:
        contrast2 = contrast1-1
    
    prev = "resnet50_eidivnorm_after_conv1_groups1_gaussianblur({},{})".format(contrast2,contrast1)
    new = "resnet50_eidivnorm_after_conv1_groups1_gaussianblur_kernel3({},{})".format(contrast2,contrast1)
    # renaming directory ''tutorialsdir"
    os.rename(prev,new)

    prev = "resnet50_divnorm_after_conv1_groups1_gaussianblur({},{})".format(contrast2,contrast1)
    new = "resnet50_divnorm_after_conv1_groups1_gaussianblur_kernel3({},{})".format(contrast2,contrast1)
    # renaming directory ''tutorialsdir"
    os.rename(prev,new)

    prev = "resnet50_divnormei_after_conv1_groups1_gaussianblur({},{})".format(contrast2,contrast1)
    new = "resnet50_divnormei_after_conv1_groups1_gaussianblur_kernel3({},{})".format(contrast2,contrast1)
    # renaming directory ''tutorialsdir"
    os.rename(prev,new)

    prev = "resnet50_gaussianblur({},{})".format(contrast2,contrast1)
    new = "resnet50_gaussianblur_kernel3({},{})".format(contrast2,contrast1)
    # renaming directory ''tutorialsdir"
    os.rename(prev,new)

print("Successfully renamed.")