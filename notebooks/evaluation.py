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

    # ------- LABELS ------  #
    # Define labels to query each image | will return a prediction for each label
    cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                        'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                        'Pneumothorax', 'Support Devices']

    # loads in ground truth labels into memory

    test_pred = np.load('/home/jex451/CheXzero/predictions/chexpert_preds.npy')
    test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

    # evaluate model, no bootstrap
    cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

    # boostrap evaluations for 95% confidence intervals
    bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

    # print results with confidence intervals
    print(bootstrap_results[1])