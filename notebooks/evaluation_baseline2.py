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
    cxr_result_path = '../results/baseline2/5_percent/cxr.csv' 
    bootstrap_result_path = '../results/baseline2/5_percent/bootstrap.csv' 

    # ------- LABELS ------  #
    # Define labels to query each image | will return a prediction for each label
    cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                        'Pneumothorax', 'Support Devices']

    # loads in ground truth labels into memory

    test_pred = np.load('/home/jex451/CheXzero/predictions/ensemble/chexpert_preds.npy')
    test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # calculate the max of all probabilties score of all classes. 
    max_probs = test_pred.max(1)  # dimension should be (500,1)

    # calculate the threshold for the lowest 10%. 
    threshold = np.percentile(max_probs, 5, axis=0)

    # Get the indices for the lowest. 
    filter_array = max_probs > threshold # dimension should be (500, 1)
    filter_array_expanded = np.repeat(filter_array, 14).reshape(len(filter_array),14) # dimension shuold be (500,14)

    # Filter out lowest ones.
    test_pred = test_pred[filter_array]
    test_true = test_true[filter_array]

    # evaluate model, no bootstrap
    cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

    # boostrap evaluations for 95% confidence intervals
    bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

    # print results with confidence intervals
    # print(bootstrap_results[1])

    # Save to csv
    cxr_results.to_csv(cxr_result_path)
    bootstrap_results[1].to_csv(bootstrap_result_path)