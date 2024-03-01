import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval


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
    cxr_filepath: str = '../data/padchest/images/2_cxr.h5' # filepath of chest x-ray images (.h5)
    cxr_true_labels_path: Optional[str] = '../data/padchest/2_cxr_labels.csv' # (optional for evaluation) if labels are provided, provide path
    model_dir: str = '../checkpoints/one_weight' # where pretrained models are saved (.pt) 
    predictions_dir: Path = Path('../predictions/padchest') # where to save predictions
    cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

    context_length: int = 77

    # ------- LABELS ------  #
    # Define labels to query each image | will return a prediction for each label
    cxr_labels: List[str] = ['aortic atheromatosis', 'central venous catheter via jugular vein', 'minor fissure thickening', 'pneumoperitoneo', 'lytic bone lesion', 'loculated pleural effusion', 'chronic changes', 'nsg tube', 'pleural thickening', 'mastectomy', 'cavitation', 'clavicle fracture', 'mammary prosthesis', 'cardiomegaly', 'hiatal hernia', 'tuberculosis sequelae', 'pulmonary mass', 'osteosynthesis material', 'kyphosis', 'artificial heart valve', 'osteopenia', 'obesity', 'sternotomy', 'suboptimal study', 'alveolar pattern', 'atelectasis', 'granuloma', 'calcified granuloma', 'reservoir central venous catheter', 'subacromial space narrowing', 'pulmonary fibrosis', 'tuberculosis', 'endotracheal tube', 'gynecomastia', 'fibrotic band', 'cervical rib', 'calcified pleural thickening', 'hydropneumothorax', 'surgery neck', 'adenopathy', 'pulmonary hypertension', 'copd signs', 'artificial mitral heart valve', 'non axial articular degenerative changes', 'fissure thickening', 'artificial aortic heart valve', 'surgery lung', 'miliary opacities', 'vertebral degenerative changes', 'single chamber device', 'air trapping', 'calcified adenopathy', 'electrical device', 'volume loss', 'central venous catheter via umbilical vein', 'azygos lobe', 'calcified pleural plaques', 'consolidation', 'vascular hilar enlargement', 'bronchovascular markings', 'descendent aortic elongation', 'hyperinflated lung', 'blastic bone lesion', 'vertebral anterior compression', 'pseudonodule', 'increased density', 'lobar atelectasis', 'bone metastasis', 'abscess', 'bullas', 'osteoporosis', 'central venous catheter', 'hilar enlargement', 'air fluid level', 'surgery breast', 'scoliosis', 'central vascular redistribution', 'reticulonodular interstitial pattern', 'end on vessel', 'subcutaneous emphysema', 'multiple nodules', 'cyst', 'flattened diaphragm', 'atelectasis basal', 'soft tissue mass', 'external foreign body', 'tracheal shift', 'goiter', 'aortic aneurysm', 'mediastinal shift', 'hypoexpansion', 'vertebral compression', 'heart insufficiency', 'aortic button enlargement', 'costochondral junction hypertrophy', 'fracture', 'axial hyperostosis', 'superior mediastinal enlargement', 'mediastinic lipomatosis', 'humeral fracture', 'right sided aortic arch', 'chilaiditi sign', 'humeral prosthesis', 'sclerotic bone lesion', 'mass', 'reticular interstitial pattern', 'aortic elongation', 'suture material', 'laminar atelectasis', 'hilar congestion', 'vertebral fracture', 'pectum carinatum', 'unchanged', 'exclude', 'calcified densities', 'pacemaker', 'mediastinal mass', 'hemidiaphragm elevation', 'dual chamber device', 'ascendent aortic elongation', 'pneumothorax', 'dai', 'hypoexpansion basal', 'metal', 'pectum excavatum', 'pulmonary edema', 'kerley lines', 'tracheostomy tube', 'callus rib fracture', 'apical pleural thickening', 'round atelectasis', 'prosthesis', 'surgery heart', 'chest drain tube', 'nodule', 'pleural effusion', 'emphysema', 'surgery', 'thoracic cage deformation', 'supra aortic elongation', 'lymphangitis carcinomatosa', 'central venous catheter via subclavian vein', 'lung metastasis', 'infiltrates', 'diaphragmatic eventration', 'mediastinal enlargement', 'ventriculoperitoneal drain tube', 'ground glass pattern', 'vascular redistribution', 'post radiotherapy changes', 'double j stent', 'bronchiectasis', 'heart valve calcified', 'costophrenic angle blunting', 'interstitial pattern', 'pneumonia', 'nipple shadow', 'rib fracture', 'normal']

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

    print(model_paths)

    predictions, y_pred_avg = ensemble_models(
        model_paths=model_paths, 
        cxr_filepath=cxr_filepath, 
        cxr_labels=cxr_labels, 
        cxr_pair_template=cxr_pair_template, 
        cache_dir=cache_dir,
    )

    # save averaged preds
    pred_name = "chexpert_preds.npy" # add name of preds
    predictions_dir = predictions_dir / pred_name
    np.save(file=predictions_dir, arr=y_pred_avg)