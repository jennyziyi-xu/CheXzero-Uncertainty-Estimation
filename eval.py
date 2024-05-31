import subprocess
import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List, Callable

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize

import sklearn
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score
from sklearn.utils import resample 

import scipy
import scipy.stats

import sys
sys.path.append('../..')

import clip
from model import CLIP

def compute_mean(stats, is_df=True): 
    spec_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    if is_df: 
        spec_df = stats[spec_labels]
        res = np.mean(spec_df.iloc[0])
    else: 
        # cis is df, within bootstrap
        vals = [stats[spec_label][0] for spec_label in spec_labels]
        res = np.mean(vals)
    return res

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    print('pred: ', pred)
    
    expand = target.expand(-1, max(topk))
    print('expand: ', expand)
    
    correct = pred.eq(expand)
    print('correct: ', correct)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def sigmoid(x): 
    z = 1/(1 + np.exp(-x)) 
    return z

def calc_TP_FP_rate(y_true, y_pred):
    # Instantiate counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            TP +=1
        elif y_true[i] == y_pred[i] == 0:
            TN +=1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN +=1
        else:
            FP +=1

    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    return tpr, fpr


def cal_auc(y_pred, y_true):
    # Containers for true positive / false positive rates
    lr_tp_rates = []
    lr_fp_rates = []

    thresholds = [1, 0.5131685,  0.51023954, 0.5095895,  0.5071519,  0.5070283,
                    0.5068729,  0.5068183,  0.50642955, 0.50640285, 0.5063635,  0.5061632,
                    0.50610185, 0.50609016, 0.5059569,  0.5059325,  0.5058838,  0.505187,
                    0.50465524, 0.50459987, 0.5045627,  0.50441366, 0.5042987,  0.50418806,
                    0.50401556, 0.50398445, 0.50371164, 0.5036885,  0.5036503,  0.50352156,
                    0.5033626,  0.5032673,  0.5031693,  0.50313556, 0.5027984,  0.50276434,
                    0.50271964, 0.50246954, 0.5023061,  0.5022084,  0.50206524, 0.5020175,
                    0.50200135, 0.50194806, 0.50181484, 0.5017765,  0.50172067, 0.50156593,
                    0.5015253,  0.5013988,  0.5013981,  0.5012277,  0.5012265,  0.501196,
                    0.5011505,  0.5009535,  0.500941,   0.50075936, 0.5007486,  0.50073254,
                    0.500667,   0.5006371,  0.5006154,  0.5002319,  0.5002295,  0.49996647,
                    0.4999562,  0.49955922, 0.49954817, 0.49940532, 0.49935192, 0.49928793,
                    0.4992445,  0.49915224, 0.49909154, 0.49897256, 0.49889764, 0.49884492,
                    0.49882773, 0.49833316, 0.49831182, 0.4978044,  0.49774432, 0.4976123,
                    0.49761084, 0.49755853, 0.49754822, 0.49752727, 0.49745739, 0.4974266,
                    0.49741152, 0.49740672, 0.49737415, 0.494697,   0.49467063, 0.48568135]

    for p in thresholds:

        y_preds_threshold = []
        for prob in y_pred:
            if prob > p:
                y_preds_threshold.append(1)
            else:
                y_preds_threshold.append(0)
    
        tp_rate, fp_rate = calc_TP_FP_rate(y_true, y_preds_threshold)
        lr_tp_rates.append(tp_rate)
        lr_fp_rates.append(fp_rate)
    
    roc_auc = auc(lr_fp_rates, lr_tp_rates)
    return roc_auc

''' ROC CURVE '''
def plot_roc(y_pred, y_true, roc_name, plot=False):
    # given the test_ground_truth, and test_predictions 
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # print(fpr, tpr)

    # plt.figure(figsize=(8, 8))
    # plt.plot(fpr, tpr)
    # plt.savefig("padchest_ensemble_scatter.png")

    roc_auc = auc(fpr, tpr)

    if plot: 
        print(thresholds)
        # print("fpr: ", fpr[:50])
        # print("tpr: ", tpr[:50])
        plt.figure(dpi=100)
        plt.title(roc_name)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plt.show()
        # plt.savefig("chexpert_ROC.png")
    return fpr, tpr, thresholds, roc_auc

# J = TP/(TP+FN) + TN/(TN+FP) - 1 = tpr - fpr
def choose_operating_point(fpr, tpr, thresholds):
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1-_fpr
            J = _tpr - _fpr
    return sens, spec

''' PRECISION-RECALL CURVE '''
def plot_pr(y_pred, y_true, pr_name, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true==1]) / len(y_true)
    
    if plot: 
        plt.figure(dpi=20)
        plt.title(pr_name)
        plt.plot(recall, precision, 'b', label='AUC = %0.2f' % pr_auc)
        # axis labels
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [baseline, baseline],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the plot
        plt.show()
    return precision, recall, thresholds

def evaluate(y_pred, y_true, cxr_labels, 
                   roc_name='Receiver Operating Characteristic', pr_name='Precision-Recall Curve', label_idx_map=None):
    
    '''
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)
    
    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1. 
    
    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr. 
    
    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity, 
    PR-Curve, Precision, Recall for each class. 
    '''
    import warnings
    warnings.filterwarnings('ignore')

    num_classes = y_pred.shape[-1] # number of total labels
    
    dataframes = []
    for i in range(num_classes): 
#         print('{}.'.format(cxr_labels[i]))

        if label_idx_map is None: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            y_true_i = y_true[:, i] # (num_samples,)
            
        else: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            
            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index] # (num_samples,)
            
        cxr_label = cxr_labels[i]
        
        ''' ROC CURVE '''
        roc_name = cxr_label + ' ROC Curve'
        # if (i==9):
        #     plot_var = True
        # else:
        #     plot_var = False
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name, plot=False)
        
        sens, spec = choose_operating_point(fpr, tpr, thresholds)

        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
        dataframes.append(df)
        
        ''' PRECISION-RECALL CURVE '''
        pr_name = cxr_label + ' Precision-Recall Curve'
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name)
        
    dfs = pd.concat(dataframes, axis=1)
    return dfs


def evaluate_custom(y_pred, y_true, cxr_labels, 
                   roc_name='Receiver Operating Characteristic', pr_name='Precision-Recall Curve', label_idx_map=None):
    
    '''
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)
    
    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1. 
    
    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr. 
    
    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity, 
    PR-Curve, Precision, Recall for each class. 
    '''
    import warnings
    warnings.filterwarnings('ignore')

    num_classes = y_pred.shape[-1] # number of total labels
    
    dataframes = []
    for i in range(num_classes): 
#         print('{}.'.format(cxr_labels[i]))

        if label_idx_map is None: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            y_true_i = y_true[:, i] # (num_samples,)
            
        else: 
            y_pred_i = y_pred[:, i] # (num_samples,)
            
            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index] # (num_samples,)
            
        cxr_label = cxr_labels[i]
        
        ''' ROC CURVE '''
        roc_name = cxr_label + ' ROC Curve'
        if (i==9):
            plot_var = True
        else:
            plot_var = False
        roc_auc = cal_auc(y_pred_i, y_true_i)
        
        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
        dataframes.append(df)
        
        # ''' PRECISION-RECALL CURVE '''
        # pr_name = cxr_label + ' Precision-Recall Curve'
        # precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name)
        
    dfs = pd.concat(dataframes, axis=1)
    return dfs


def evaluate_by_class(y_pred, y_true, cxr_labels, i,
                   roc_name='Receiver Operating Characteristic', pr_name='Precision-Recall Curve', label_idx_map=None):
    
    '''
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples)
    
    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1. 
    
    `y_true` is a numpy array consisting of binary values representing if a class is present in
    the cxr. 
    
    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity, 
    PR-Curve, Precision, Recall for each class. 
    '''
    import warnings
    warnings.filterwarnings('ignore')
    
    dataframes = []
    # for i in range(num_classes): 
    print('{}.'.format(cxr_labels[i]))
        
    cxr_label = cxr_labels[i]
    
    ''' ROC CURVE '''
    roc_name = cxr_label + ' ROC Curve'
    fpr, tpr, thresholds, roc_auc = plot_roc(y_pred, y_true, roc_name)
    
    sens, spec = choose_operating_point(fpr, tpr, thresholds)

    results = [[roc_auc]]
    df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
    dataframes.append(df)
    
    ''' PRECISION-RECALL CURVE '''
    pr_name = cxr_label + ' Precision-Recall Curve'
    precision, recall, thresholds = plot_pr(y_pred, y_true, pr_name)
    # print("precision:", precision)
        
    dfs = pd.concat(dataframes, axis=1)
    return dfs

''' Bootstrap and Confidence Intervals '''
def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df
    
def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000, label_idx_map=None): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    
    Note: 
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested
    '''
    np.random.seed(97)
    y_pred # (500, n_total_labels)
    y_true # (500, n_cxr_labels) 
    
    idx = np.arange(len(y_true))
    
    boot_stats = []
    for i in range(n_samples): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        sample_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    boot_stats = pd.concat(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)

def bootstrap_custom(y_pred, y_true, cxr_labels, n_samples=1000, label_idx_map=None): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    
    Note: 
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested
    '''
    np.random.seed(97)
    y_pred # (500, n_total_labels)
    y_true # (500, n_cxr_labels) 
    
    idx = np.arange(len(y_true))
    
    boot_values = []
    for i in range(n_samples): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        result = evaluate(y_pred_sample, y_true_sample, cxr_labels, label_idx_map=label_idx_map)
        # we know that cxr_labels is a list of length 0. 
        boot_values.append(result.at[0, cxr_labels[0] + '_auc'])

    return boot_values