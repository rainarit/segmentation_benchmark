import numpy as np
import os
import argparse
import natsort 
from csv import reader

parser = argparse.ArgumentParser(description='Organization')
parser.add_argument('--dataset', default='imagenetc')
parser.add_argument('--model', default='vgg9_ctrl')

def main():
    args = parser.parse_args()
    # output_name = args.model + '_'+ args.dataset +'_' + str(args.pertubation) + '_100'
    # output_name = os.path.join('/home/AD/rraina/segmentation_benchmark/semseg/outputs', output_name)
    # os.mkdir(output_name)

    # acc_path = os.path.join(output_name, args.model + '_imagenetc_' + str(args.pertubation) + '_100.csv')

    model_names = ['vgg9_ctrl', 'vgg9_eidovnorm_mini', 'vgg9_dalernn_mini']
    
    distortions = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 
                   'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 
                   'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
    attacks = ['LinfFastGradientAttack']
    output_folder = '/home/AD/rraina/segmentation_benchmark/semseg/outputs'
    output_list = os.listdir(output_folder)

    if args.dataset == 'imagenetc':
        for pertubation in distortions:
            final_file = os.path.join(output_folder, args.model + '_' + args.dataset + '_' + pertubation + '.csv')
            print(final_file)
            acc_list = []
            for output in natsort.natsorted(output_list):
                if pertubation in output and args.model in output:
                    pertub_file = output_folder + '/' + output + '/' + output + '.csv'
                    with open(pertub_file, 'r') as read_obj:
                        csv_reader = reader(read_obj)
                        for row in csv_reader:
                            acc_list.append(row)
            with open(final_file, 'w') as f:
                for item in acc_list:
                    f.write("%s\n" % item)   
    elif args.dataset ==  'foolbox':
        for attack in attacks:
            final_file = os.path.join(output_folder, args.model + '_' + args.dataset + '_' + attack + '_imagenet100' + '.csv')
            print(final_file)
            acc_list = []
            for output in natsort.natsorted(output_list):
                if args.dataset in output and args.model in output and 'csv' not in output:
                    pertub_file = output_folder + '/' + output + '/' + attack + '.txt'
                    with open(pertub_file, 'r') as read_obj:
                        csv_reader = reader(read_obj)
                        for i, row in enumerate(csv_reader):
                            if i > 1:
                                acc_list.append(float(row[0].split(': ')[1][:-2]))
            with open(final_file, 'w') as f:
                for item in acc_list:
                    f.write("%s\n" % item)
    return

if __name__ == '__main__':
    main()