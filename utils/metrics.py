import numpy as np
import torch
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.optimize import brentq

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    acc = float(tp +tn ) /dist.shape[0]
    return acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn


def HTER(threshold, dist, actual_issame):
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
    far = 0 if (fp==0) else float(fp) / float(tn +fp)
    frr = 0 if (fn==0) else float(fn) / float(fn +tp)
    hter = (far + frr) / 2.0
    return hter

def eer_auc(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    return eer, AUC

def get_metrics(pred, gt, thre):
    gt_bool = (gt>thre)
    acc = calculate_accuracy(thre, pred, gt_bool)
    hter = HTER(thre, pred, gt_bool)
    eer, auc = eer_auc(gt, pred) 
    return acc, auc, hter, eer