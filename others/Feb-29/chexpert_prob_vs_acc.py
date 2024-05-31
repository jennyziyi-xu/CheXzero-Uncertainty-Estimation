import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('../../')
import zero_shot
import eval

# /home/jex451/.conda/envs/chexzero2/bin/python chexpert_prob_vs_acc.py --batch_size=50 "Pleural Effusion" "Atelectasis" "Cardiomegaly" "Consolidation" "Edema"


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--plot', type=bool)
    parser.add_argument('condition', nargs='*', help="Pleural Effusion, etc. .")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    batch_size = args.batch_size
    cxr_labels_original: List[str] = ['Atelectasis','Cardiomegaly', 
                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                            'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                            'Pneumothorax', 'Support Devices']

    # Load in all predictions from one checkpoint. 
    test_pred = np.load('/home/jex451/CheXzero/predictions/ensemble/chexpert_preds.npy')

    # breakpoint()

    for condition in args.condition:

        print(condition)

        # Get for only the class 
        condition_index = cxr_labels_original.index(condition)

        test_pred_condition = test_pred[:,condition_index]

        # Load in all true labels, filter for only the class "pleural effusion"

        cxr_true_labels_path: Optional[str] = '/home/jex451/CheXzero/data/groundtruth.csv'
        cxr_labels = [condition]
        test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

        # Sort according to prediction probabilities. 
        both_lists = zip(test_pred_condition, test_true)
        both_lists_sorted = sorted(both_lists, key=lambda x:x[0])

        condition_probs = [val for (val, _) in both_lists_sorted]
        test_true_sorted = [val for (_, val) in both_lists_sorted]

        n = len(condition_probs)
        test_pred_condition_sorted = np.array(condition_probs).reshape(n, 1)
        test_true_sorted = np.array(test_true_sorted).reshape(n, 1)

        # Calculate AUC score for slices 
        y_axis = []
        x_axis = []

        slices = len(test_true_sorted)//batch_size
        
        for i in range(slices):
            probs_slice = test_pred_condition_sorted[i*batch_size: (i+1)*batch_size]
            test_slice = test_true_sorted[i*batch_size: (i+1)*batch_size]
            x_axis.append(np.average(probs_slice))
            cxr_results: pd.DataFrame = eval.evaluate(probs_slice, test_slice, cxr_labels)
            y_axis.append(cxr_results.at[0, condition+'_auc'])
        
        print(y_axis)
        for i in range(len(y_axis)):
            if np.isnan(y_axis[i]):
                y_axis[i] = 0
        
        print(y_axis)

        if (args.plot):
            plt.figure()
            plt.xlabel("Mean probability for the batch")
            plt.ylabel("batch AUC")
            plt.title("chexpert- {} \n -Relationship between probabilities and AUC in batches of {}".format(condition, args.batch_size))
            # plt.plot(x_axis, y_axis)
            plt.scatter(x_axis, y_axis)
            plt.savefig("chexpert3_{}_probs_vs_auc.png".format(condition))
        