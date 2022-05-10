#!/usr/bin/python

import os
import sys

occlude_range = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]

dalernn_models = ['uszrla']
base_models = ['dlkoiv', 'cvhwov']
eidiv_models = ['sxphdx', 'tvdydh']

for o_r in occlude_range:
	# for model in dalernn_models:
	# 	model_name = "vgg9_dalernn_mini"
	# 	command = "python backbone_eval_vgg_occlusion.py -a %s -ol %s -oh %s /mnt/cube/projects/imagenet_100 --batch-size 512 --print-freq 1 --gpu 1 --resume /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/%s/checkpoint_best_%s.pth" % (model_name, o_r[0], o_r[1], model, model_name)
	# 	print(command)
	# 	os.system(command)

	for model in eidiv_models:
		model_name = "vgg9_eidivnorm_mini"
		command = "python backbone_eval_vgg_occlusion.py -a %s -ol %s -oh %s /mnt/cube/projects/imagenet_100 --batch-size 512 --print-freq 1 --gpu 1 --resume /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/%s/checkpoint_best_%s.pth" % (model_name, o_r[0], o_r[1], model, model_name)
		print(command)
		os.system(command)

	for model in base_models:
		model_name = "vgg9_ctrl"
		command = "python backbone_eval_vgg_occlusion.py -a %s -ol %s -oh %s /mnt/cube/projects/imagenet_100 --batch-size 512 --print-freq 1 --gpu 1 --resume /home/vveeraba/src/segmentation_benchmark/semseg/checkpoints_imagenet_100/%s/checkpoint_best_%s.pth" % (model_name, o_r[0], o_r[1], model, model_name)
		print(command)
		os.system(command)