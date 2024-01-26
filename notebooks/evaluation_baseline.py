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
    cxr_result_path = '../results/baseline/20_percent/cxr.csv'
    bootstrap_result_path = '../results/baseline/20_percent/bootstrap.csv'

    # ------- LABELS ------  #
    # Define labels to query each image | will return a prediction for each label
    cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                        'Pneumothorax', 'Support Devices']

    # loads in ground truth labels into memory

    test_pred = np.load('/home/jex451/CheXzero/predictions/ensemble/chexpert_preds.npy')
    test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    threshold = np.percentile(test_pred, 20, axis=0)    # filter out 10% of lowest uncertainties. 
    # filter out both arrays based on this threshold
    # print(threshold)
    num_classes =len(threshold)
    # num_classes = 1
    dataframes_cxr = []
    dataframes_bootstrap = []
    for i in range(num_classes):
        test_pred_i = test_pred[:,i]
        test_true_i = test_true[:,i]
        # print("threshold", cxr_labels[i], threshold[i])
        filter_array = test_pred_i > threshold[i]
        # # debugging 
        # # indices = np.argwhere(filter_array == False)
        # # print(np.take(test_true_i, indices))
        
        test_pred_i_filterd = test_pred_i[filter_array].reshape(-1,1)
        print(test_pred_i_filterd.shape)
        test_true_i_filterd = test_true_i[filter_array].reshape(-1,1)
        
        cxr_result = eval.evaluate_by_class(test_pred_i_filterd, test_true_i_filterd, cxr_labels,i)
        dataframes_cxr.append(cxr_result)

        bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred_i_filterd, test_true_i_filterd, np.array([cxr_labels[i]])) # (df of results for each bootstrap, df of CI)
        dataframes_bootstrap.append(bootstrap_results[1])
    
    cxr_result_concatenated = pd.concat(dataframes_cxr, axis=1)
    bootstrap_result_concatenated = pd.concat(dataframes_bootstrap, axis=1)
   
    cxr_result_concatenated.to_csv(cxr_result_path)
    bootstrap_result_concatenated.to_csv(bootstrap_result_path)

