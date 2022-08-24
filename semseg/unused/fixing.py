import numpy as np

for e in range(0, 50):
    path = "/home/AD/rraina/segmentation_benchmark/semseg/results/resnet50_eidivnorm_deeplabv3/zfbtdy/epoch_{}/image_mean_ious.csv".format(e)
    f1 = open(path, "r")
    lines = f1.readlines()
    for l in lines:
        print(l)
    #str_list = last_line.split(",")
    #float_list = [float(x) for x in str_list]
    #print(np.mean(float_list))
    f1.close()