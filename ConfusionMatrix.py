from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
import torch.utils.data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision
from torch import nn
import numpy as np
from PIL import Image
import random
import errno
import os
import pathlib
from utils import nanmean

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def eval_metrics(self):
        hist = self.mat
        overall_accuracy = overall_pixel_accuracy(hist)
        per_class_accuracy = per_class_pixel_accuracy(hist)
        iou = jaccard_index(hist)
        dice = dice_coefficient(hist)
        return overall_accuracy, per_class_accuracy, iou, dice


    def overall_pixel_accuracy(self, hist):
        """Computes the total pixel accuracy.
        The overall pixel accuracy provides an intuitive
        approximation for the qualitative perception of the
        label when it is viewed in its overall shape but not
        its details.
        Args:
            hist: confusion matrix.
        Returns:
            overall_acc: the overall pixel accuracy.
        """
        correct = torch.diag(hist).sum()
        total = hist.sum()
        overall_acc = correct / (total)
        return overall_acc

    def per_class_pixel_accuracy(self, hist):
        """Computes the average per-class pixel accuracy.
        The per-class pixel accuracy is a more fine-grained
        version of the overall pixel accuracy. A model could
        score a relatively high overall pixel accuracy by
        correctly predicting the dominant labels or areas
        in the image whilst incorrectly predicting the
        possibly more important/rare labels. Such a model
        will score a low per-class pixel accuracy.
        Args:
            hist: confusion matrix.
        Returns:
            avg_per_class_acc: the average per-class pixel accuracy.
        """
        correct_per_class = torch.diag(hist)
        total_per_class = hist.sum(dim=1)
        per_class_acc = correct_per_class / (total_per_class)
        avg_per_class_acc = nanmean(per_class_acc)
        return avg_per_class_acc

    def jaccard_index(self, hist):
        """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).
        Args:
            hist: confusion matrix.
        Returns:
            avg_jacc: the average per-class jaccard index.
        """
        A_inter_B = torch.diag(hist)
        A = hist.sum(dim=1)
        B = hist.sum(dim=0)
        jaccard = A_inter_B / (A + B - A_inter_B)
        avg_jacc = nanmean(jaccard)
        return avg_jacc

    def dice_coefficient(self, hist):
        """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
        Args:
            hist: confusion matrix.
        Returns:
            avg_dice: the average per-class dice coefficient.
        """
        A_inter_B = torch.diag(hist)
        A = hist.sum(dim=1)
        B = hist.sum(dim=0)
        dice = (2 * A_inter_B) / (A + B)
        avg_dice = nanmean(dice)
        return avg_dice

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)
    
    def __str__(self):
        acc_global, acc, iou, dice = self.eval_metrics()
        return (
            'global pixel accuracy: {:.1f}\n'
            'class pixel accuracy: {}\n'
            'global IoU: {:.1f}\n'
            'class IoU: {}\n'
            'global dice coefficient: {:.1f}\n'
            'class dice coefficient: {}\n'
        ).format(
            acc_global.item()*100,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            iou.mean().item() * 100,
            ['{:.1f}'.format(i) for i in (iou * 100).tolist()],
            dice.mean().item() * 100,
            ['{:.1f}'.format(i) for i in (dice * 100).tolist()]
        )

    #def __str__(self):
    #    acc_global, acc, iu = self.compute()
    #    return (
    #        'global correct: {:.1f}\n'
    #        'average row correct: {}\n'
    #        'IoU: {}\n'
    #        'mean IoU: {:.1f}').format(
    #            acc_global.item() * 100,
    #            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
    #            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
    #            iu.mean().item() * 100)