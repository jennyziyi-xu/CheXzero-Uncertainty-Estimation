import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional


import sys
sys.path.append('../')
import zero_shot
import eval



if __name__ == "__main__":

    cxr_true_labels_path: Optional[str] = '../data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
    cxr_result_path = '../results/mc/dropout/cxr.csv' 
    bootstrap_result_path = '../results/mc/dropout/bootstrap.csv' 

    # ------- LABELS ------  #
    # Define labels to query each image | will return a prediction for each label
    cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                        'Pneumothorax', 'Support Devices']

    
    # Load all predictions from the 10 experiments here
    test_pred_1 = np.load('/home/jex451/CheXzero/predictions/mc_drop_1/cached/best_64_0.0001_original_16000_0.861.npy')

    # print(test_pred_1.shape)
    test_pred_2 = np.load('/home/jex451/CheXzero/predictions/mc_drop_2/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_3 = np.load('/home/jex451/CheXzero/predictions/mc_drop_3/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_4 = np.load('/home/jex451/CheXzero/predictions/mc_drop_4/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_5 = np.load('/home/jex451/CheXzero/predictions/mc_drop_5/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_6 = np.load('/home/jex451/CheXzero/predictions/mc_drop_6/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_7 = np.load('/home/jex451/CheXzero/predictions/mc_drop_7/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_8 = np.load('/home/jex451/CheXzero/predictions/mc_drop_8/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_9 = np.load('/home/jex451/CheXzero/predictions/mc_drop_9/cached/best_64_0.0001_original_16000_0.861.npy')
    test_pred_10 = np.load('/home/jex451/CheXzero/predictions/mc_drop_10/cached/best_64_0.0001_original_16000_0.861.npy')

    n,m = test_pred_1.shape

    # Concatenate all 10 arrays to produce (10, 500, 14) array
    all_preds = np.concatenate((test_pred_1.reshape((1,n,m)), test_pred_2.reshape((1,n,m)), test_pred_3.reshape((1,n,m)), test_pred_4.reshape((1,n,m)), test_pred_5.reshape((1,n,m)), 
                            test_pred_6.reshape((1,n,m)), test_pred_7.reshape((1,n,m)), test_pred_8.reshape((1,n,m)), test_pred_9.reshape((1,n,m)), test_pred_10.reshape((1,n,m))),axis=0)
    

    # # Compute Variance from 10 experiments for all_preds. 

    std = np.std(all_preds, axis=0)   # dimension should be (500, 14)
    # print(variance.shape)

    # # Get the max variance among 14 predictions 
    max_std = np.max(std, axis=1)    # dimension should be (500,1)
    # print(max_variance.shape)

    # filter out 15% of samples with highest variance 
    threshold = np.percentile(max_std, 85)
    # print(threshold) 0.002669
    # get indices for the 85% lower variance 
    filter_array = max_std < threshold

    test_pred = np.load('/home/jex451/CheXzero/predictions/ensemble/cached/best_64_0.0001_original_16000_0.861.npy')

    test_pred = test_pred[filter_array]

    test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    test_true = test_true[filter_array]

    # # evaluate model, no bootstrap
    cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

    # # boostrap evaluations for 95% confidence intervals
    bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

    # print results with confidence intervals
    # print(bootstrap_results[1])

    # Save to csv
    cxr_results.to_csv(cxr_result_path)
    bootstrap_results[1].to_csv(bootstrap_result_path)