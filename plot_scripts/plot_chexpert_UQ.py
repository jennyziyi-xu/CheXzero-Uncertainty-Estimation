# Plot the result of applying three different uncertainty estimation methods on the Chexpert dataset.
# 1, Median Removal 
# 2, MSP
# 3, Monte Carlo Dropout
# Plots will be saved in the current directory

import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import sys
sys.path.append('../../')
import zero_shot
import eval

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot the results of applying three uncertainty estimation techniques.")
    parser.add_argument("--condition", type=str, required=True, help="one of ['Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema']")
    parser.add_argument("--gt_path", type=str, required=True, help="The path to the ground truth labels csv file.")
    parser.add_argument("--pred_path", type=str, required=True, help="The path to the predictions.")
    parser.add_argument("--MCD_pred_path", type=str, required=True, help="The path to the MCD predictions.")
    args = parser.parse_args()
    
    disease = args.condition

    # Load in all predictions from one checkpoint. 
    test_pred = np.load(args.pred_path)
    cxr_labels_all: List[str] = ['Atelectasis','Cardiomegaly', 
                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                            'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                            'Pneumothorax', 'Support Devices']

    disease_index = cxr_labels_all.index(disease)

    # Get for only the class (disease)
    test_pred_disease = test_pred[:,disease_index]
    total_samples = len(test_pred_disease)

    # Load in all true labels, filter for only the class (disease)
    test_true_disease = zero_shot.make_true_labels(cxr_true_labels_path=args.gt_path, cxr_labels=[disease])

    # create the x-axis, from 0 to 50 with step size 2.
    x_axis = np.arange(0, int(0.5 * total_samples), 2) / 5
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(5, 6))   

    ##############################################
    ## 1, Median removal 
    
    # Sort probabilities in terms of their probabilities, keep the indices. 
    test_pred_dict= dict(enumerate(test_pred_disease))
    test_pred_sorted = sorted(test_pred_dict.items(), key=lambda test_pred_dict: test_pred_dict[1])

    auc_array= []
    middle_index = total_samples // 2
    i=0
    while i < 0.5 * total_samples:
        # remove the middle elements. Calculate the AUC for the non-removed part. 
        remove_indices = [x[0] for x in test_pred_sorted[(middle_index - i//2) : middle_index + i//2]]

        new_test_pred = np.delete(test_pred_disease, remove_indices)
        new_test_true = np.delete(test_true_disease, remove_indices)

        # using 1 evaluation
        result = eval.evaluate(new_test_pred.reshape(-1, 1), new_test_true.reshape(-1,1), [disease])
        auc_array.append(result.at[0, disease+'_auc'])

        i+=2

    sns.lineplot(x = x_axis, y= auc_array, color='blue', label='Median Removal')

    ##################################################
    ## 2, MSP

    # calculate the max of all probabilties score of all classes. 
    max_probs = test_pred.max(1) 
    auc_array = []
    for t in range(0, 51, 2):
        # calculate the threshold for the lowest t%. 
        threshold = np.percentile(max_probs, t, axis=0)
        filter_array = max_probs > threshold

        # filter out lowest ones
        test_pred_msp = test_pred_disease[filter_array]
        test_true_msp = test_true_disease[filter_array]

        # evaluate model 
        result = eval.evaluate(test_pred_msp.reshape(-1, 1), test_true_msp.reshape(-1,1), [disease])
        auc_array.append(result.at[0, disease+'_auc'])
    
    x_axis = np.arange(0,51,2)
    sns.lineplot(x = x_axis, y= auc_array, color='red', label='MSP')


    ##################################################
    ## 3, Monte Carlo Dropout

    pred_dir = args.MCD_pred_path
    all_preds = np.array([])
    for i in range(1, 31):
        test_pred = np.load(pred_dir + "chexpert_preds_{}.npy".format(i))
        n,m  = test_pred.shape
        if len(all_preds) == 0:
            all_preds = test_pred.reshape((1,n,m))
        else:
            all_preds = np.concatenate((all_preds, test_pred.reshape((1,n,m))), axis=0)
      
    # compute variance
    std_all = np.std(all_preds, axis=0)
    std_disease = std_all[:, disease_index]

    auc_array = []

    for t in range(0, 51, 2):
        # filter out t% of samples with highest variance 
        threshold = np.percentile(std_disease, 100 - t)
        filter_array = std_disease < threshold

        # filter out highest variance
        mc_pred = test_pred_disease[filter_array]
        mc_true = test_true_disease[filter_array]
        # evaluate model 
        result = eval.evaluate(mc_pred.reshape(-1, 1), mc_true.reshape(-1,1), [disease])
        auc_array.append(result.at[0, disease+'_auc'])

    x_axis = np.arange(0,51,2)
    mcd_plot = sns.lineplot(x = x_axis, y= auc_array, color="green", label='MCD')
    mcd_plot.legend_.remove()

    plt.title(disease + " - CheXpert")
    plt.savefig(disease+"_chexpert.png")