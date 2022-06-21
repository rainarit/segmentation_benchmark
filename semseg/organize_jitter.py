from re import A
import numpy as np
import os
import argparse
import natsort 
from csv import reader

parser = argparse.ArgumentParser(description='Organization')
parser.add_argument('--model', default='vgg9_ctrl')

def main():
    args = parser.parse_args()
    # output_name = args.model + '_'+ args.dataset +'_' + str(args.pertubation) + '_100'
    # output_name = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/outputs', output_name)
    # os.mkdir(output_name)

    # acc_path = os.path.join(output_name, args.model + '_imagenetc_' + str(args.pertubation) + '_100.csv')

    model_names = ['deeplabv3_resnet50', 'deeplabv3_eidivnorm_resnet50', 'vgg9_ctrl', 'vgg9_dalernn_mini', 'vgg9_eidivnorm_mini']
    
    #distortions = ['brightness', 'contrast', 'sigma']
    distortions = ['occlusion']
    seeds = ['', '429']
    attacks = ['LinfFastGradientAttack']
    output_folder = '/home/AD/rraina/segmentation_benchmark/semseg/outputs'
    output_list = os.listdir(output_folder)

    for pertubation in distortions:
        for model in model_names:
            if 'vgg' in model:
                acc_list = []
                final_file = os.path.join(output_folder, model + '_imagenet_100_' + pertubation + '.csv')
                for a in range(0, 11, 1):
                    a = a/10.0
                    if a == 0:
                        b = a 
                    else:
                        b = a - 0.1
                    if b == 0.0 and a != 0:
                        b = 0
                        a = round(a, 1)
                    else:
                        b = round(b, 1)
                        b = str(b)[1:3]
                        if b == '.0' and a == 0:
                            b = '0.0'
                            a = '0.0'
                    file = '{}_imagenet_100_{}_{}_{}.csv'.format(model, pertubation, b, a)
                    path = os.path.join(output_folder, file)
                    with open(path, 'r') as read_obj:
                        csv_reader = reader(read_obj)
                        for row in csv_reader:
                            if 'val_accuracy1' in row:
                                acc_list.append(float(row[1]))
                           
                    with open(final_file, 'w') as f:
                        for item in acc_list:
                            f.write("%s\n" % item)







   # brightness_(1,1)/val_test_only/mean_iou.csv

    # for pertubation in distortions:
    #     for model in model_names:
    #         for seed in seeds:
    #             acc_list = []
    #             final_file = os.path.join(output_folder, model + '_' + pertubation + seed +'.csv')
    #             print(final_file)
    #             for a in range(0, 11, 1):
    #                 a = a/10.0
    #                 if a == 0:
    #                     b = a 
    #                 else:
    #                     b = a - 0.1
    #                 if b == 0.0 and a != 0:
    #                     b = 0
    #                     a = round(a, 1)
    #                 else:
    #                     b = round(b, 1)
    #                     b = str(b)[1:3]
    #                     if b == '.0' and a == 0:
    #                         b = '0.0'
    #                         a = '0.0'
    #                 if seed != '':
    #                     file = '{}_{}_({},{})/{}/val_test_only/mean_iou.csv'.format(model, pertubation, b, a, seed)
    #                 else:
    #                     file = '{}_{}_({},{})/val_test_only/mean_iou.csv'.format(model, pertubation, b, a)
    #                 path = os.path.join(output_folder, file)
    #                 with open(path, 'r') as read_obj:
    #                     csv_reader = reader(read_obj)
    #                     for row in csv_reader:
    #                         acc_list.append(float(row[0]))
                        
    #                 with open(final_file, 'w') as f:
    #                     for item in acc_list:
    #                         f.write("%s\n" % item) 



    # for pertubation in distortions:
    #     for model in model_names:
    #         acc_list = []
    #         final_file = os.path.join(output_folder, model + '_' + pertubation + '.csv')
    #         for a in range(1, 21, 1):
    #             if a == 1:
    #                 b = a 
    #             else:
    #                 b = a - 1
    #             file = '{}_{}_({},{})/val_test_only/mean_iou.csv'.format(model, pertubation, b, a)
    #             path = os.path.join(output_folder, file)
    #             with open(path, 'r') as read_obj:
    #                 csv_reader = reader(read_obj)
    #                 for row in csv_reader:
    #                     acc_list.append(float(row[0]))

    #             with open(final_file, 'w') as f:
    #                 for item in acc_list:
    #                     f.write("%s\n" % item) 

    # for model in model_names:
    #     acc_list = []
    #     path = os.path.join(output_folder, model + '/' + 'eval_mean_iou.csv')
    #     final_file = os.path.join(output_folder, model + '.csv')

    #     with open(path, 'r') as read_obj:
    #         csv_reader = reader(read_obj)
    #         for row in csv_reader:
    #             acc_list.append(float(row[0]))

    #     with open(final_file, 'w') as f:
    #         for item in acc_list:
    #             f.write("%s\n" % item) 

if __name__ == '__main__':
    main()