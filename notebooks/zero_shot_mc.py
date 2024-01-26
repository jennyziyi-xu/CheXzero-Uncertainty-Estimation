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
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

## Run the model on the data set using ensembled models
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

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 

        # Try 1
        # module = model.visual.conv1
        # prune.random_unstructured(module, name="weight", amount=0.1)
        
        # Try 2 
        for i in range(12):
            prune.random_unstructured(model.visual.transformer.resblocks[i].attn.out_proj, name="weight", amount=0.15)

        for j in range(12):
            prune.random_unstructured(model.transformer.resblocks[j].attn.out_proj, name="weight", amount=0.15)
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else: # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg

## Define Zero Shot Labels and Templates

if __name__ == "__main__":

    # ----- DIRECTORIES ------ #
    cxr_filepath: str = '../data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
    cxr_true_labels_path: Optional[str] = '../data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
    model_dir: str = '../checkpoints/one_weight' # where pretrained models are saved (.pt) 
    predictions_dir: Path = Path('../predictions/mc_drop_10') # where to save predictions
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

    # print(model_paths)

    predictions, y_pred_avg = ensemble_models(
        model_paths=model_paths, 
        cxr_filepath=cxr_filepath, 
        cxr_labels=cxr_labels, 
        cxr_pair_template=cxr_pair_template, 
        cache_dir=cache_dir,
    )

    print(y_pred_avg)

    # save averaged preds
    # pred_name = "chexpert_preds.npy" # add name of preds
    # predictions_dir = predictions_dir / pred_name
    # np.save(file=predictions_dir, arr=y_pred_avg)