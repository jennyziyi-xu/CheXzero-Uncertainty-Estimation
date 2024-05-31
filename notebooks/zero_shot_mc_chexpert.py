import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

import torch.nn.utils.prune as prune

def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths) # ensure consistency of 
    for path in model_paths: # for each model
        model_name = Path(path).stem

        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
    
        for i in range(12):
            prune.random_unstructured(model.visual.transformer.resblocks[i].attn.out_proj, name="weight", amount=0.15)

        for j in range(12):
            prune.random_unstructured(model.transformer.resblocks[j].attn.out_proj, name="weight", amount=0.15)
    

        # if prediction already cached, don't recompute prediction
        print("Inferring model {}".format(path))
        y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

## Define Zero Shot Labels and Templates

if __name__ == "__main__":

    for iter in range(1, 31):

        # ----- DIRECTORIES ------ #
        cxr_filepath: str = '../data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
        cxr_true_labels_path: Optional[str] = '../data/groundtruth.csv'# (optional for evaluation) if labels are provided, provide path
        model_dir: str = '../checkpoints/one_weight' # where pretrained models are saved (.pt) 
        predictions_dir: Path = Path('../predictions/chexpert_MCD') # where to save predictions
        cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

        context_length: int = 77

        # ------- LABELS ------  #
        # Define labels to query each image | will return a prediction for each label
        cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                            'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                            'Pneumothorax', 'Support Devices']

        # ---- TEMPLATES ----- # 
        # Define set of templates | see Figure 1 for more details                        
        cxr_pair_template: Tuple[str] = ("{}", "no {}")

        # ----- MODEL PATHS ------ #
        # If using ensemble, collect all model paths
        model_paths = []
        for subdir, dirs, files in os.walk(model_dir):
            for file in files:
                full_dir = os.path.join(subdir, file)
                model_paths.append(full_dir)

        predictions, y_pred_avg = ensemble_models(
            model_paths=model_paths, 
            cxr_filepath=cxr_filepath, 
            cxr_labels=cxr_labels, 
            cxr_pair_template=cxr_pair_template, 
            cache_dir=cache_dir,
        )

        # save averaged preds
        pred_name = "chexpert_preds_{}.npy".format(iter) # add name of preds
        predictions_dir = predictions_dir / pred_name
        np.save(file=predictions_dir, arr=y_pred_avg)
        print("iteration done: ", iter)