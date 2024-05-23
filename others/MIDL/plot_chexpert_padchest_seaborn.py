# Make a plot where the x-axis is the percetange of samples removed from the middle. The y axis is the AUC for the 
# non-removed part. 

import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../../')
import zero_shot
import eval

if __name__ == "__main__":

    # Load in all predictions from one checkpoint. 
    test_pred_chexpert = np.load('/home/jex451/CheXzero/predictions/ensemble/chexpert_preds.npy')


    test_pred_padchest = np.load('/home/jex451/CheXzero/predictions/padchest_ensemble/padchest_preds.npy')

    cxr_labels_all_chexpert: List[str] = ['Atelectasis','Cardiomegaly', 
                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                            'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                            'Pneumothorax', 'Support Devices']

    cxr_labels_all_padchest: List[str] = ['aortic atheromatosis', 'central venous catheter via jugular vein', 'minor fissure thickening', 'pneumoperitoneo', 'lytic bone lesion', 'loculated pleural effusion', 'chronic changes', 'nsg tube', 'pleural thickening', 'mastectomy', 'cavitation', 'clavicle fracture', 'mammary prosthesis', 'cardiomegaly', 'hiatal hernia', 'tuberculosis sequelae', 'pulmonary mass', 'osteosynthesis material', 'kyphosis', 'artificial heart valve', 'osteopenia', 'obesity', 'sternotomy', 'suboptimal study', 'alveolar pattern', 'atelectasis', 'granuloma', 'calcified granuloma', 'reservoir central venous catheter', 'subacromial space narrowing', 'pulmonary fibrosis', 'tuberculosis', 'endotracheal tube', 'gynecomastia', 'fibrotic band', 'cervical rib', 'calcified pleural thickening', 'hydropneumothorax', 'surgery neck', 'adenopathy', 'pulmonary hypertension', 'copd signs', 'artificial mitral heart valve', 'non axial articular degenerative changes', 'fissure thickening', 'artificial aortic heart valve', 'surgery lung', 'miliary opacities', 'vertebral degenerative changes', 'single chamber device', 'air trapping', 'calcified adenopathy', 'electrical device', 'volume loss', 'central venous catheter via umbilical vein', 'azygos lobe', 'calcified pleural plaques', 'consolidation', 'vascular hilar enlargement', 'bronchovascular markings', 'descendent aortic elongation', 'hyperinflated lung', 'blastic bone lesion', 'vertebral anterior compression', 'pseudonodule', 'increased density', 'lobar atelectasis', 'bone metastasis', 'abscess', 'bullas', 'osteoporosis', 'central venous catheter', 'hilar enlargement', 'air fluid level', 'surgery breast', 'scoliosis', 'central vascular redistribution', 'reticulonodular interstitial pattern', 'end on vessel', 'subcutaneous emphysema', 'multiple nodules', 'cyst', 'flattened diaphragm', 'atelectasis basal', 'soft tissue mass', 'external foreign body', 'tracheal shift', 'goiter', 'aortic aneurysm', 'mediastinal shift', 'hypoexpansion', 'vertebral compression', 'heart insufficiency', 'aortic button enlargement', 'costochondral junction hypertrophy', 'fracture', 'axial hyperostosis', 'superior mediastinal enlargement', 'mediastinic lipomatosis', 'humeral fracture', 'right sided aortic arch', 'chilaiditi sign', 'humeral prosthesis', 'sclerotic bone lesion', 'mass', 'reticular interstitial pattern', 'aortic elongation', 'suture material', 'laminar atelectasis', 'hilar congestion', 'vertebral fracture', 'pectum carinatum', 'unchanged', 'exclude', 'calcified densities', 'pacemaker', 'mediastinal mass', 'hemidiaphragm elevation', 'dual chamber device', 'ascendent aortic elongation', 'pneumothorax', 'dai', 'hypoexpansion basal', 'metal', 'pectum excavatum', 'pulmonary edema', 'kerley lines', 'tracheostomy tube', 'callus rib fracture', 'apical pleural thickening', 'round atelectasis', 'prosthesis', 'surgery heart', 'chest drain tube', 'nodule', 'pleural effusion', 'emphysema', 'surgery', 'thoracic cage deformation', 'supra aortic elongation', 'lymphangitis carcinomatosa', 'central venous catheter via subclavian vein', 'lung metastasis', 'infiltrates', 'diaphragmatic eventration', 'mediastinal enlargement', 'ventriculoperitoneal drain tube', 'ground glass pattern', 'vascular redistribution', 'post radiotherapy changes', 'double j stent', 'bronchiectasis', 'heart valve calcified', 'costophrenic angle blunting', 'interstitial pattern', 'pneumonia', 'nipple shadow', 'rib fracture', 'normal']

    cxr_true_labels_path_padchest: Optional[str] = '../../data/padchest/2_cxr_labels.csv'

    cxr_true_labels_path_chexpert: Optional[str] = '/home/jex451/CheXzero/data/groundtruth.csv'

    total_samples = 500
    x_axis_chexpert = np.arange(0,int(0.8 * total_samples),2) / 5
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    

    # all_diseases_chexpert  = ['Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema']
    # all_diseases_padchest  = ['pleural effusion', 'atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema']

    disease_chexpert = 'Edema'
    disease_padchest  = 'pulmonary edema'

    disease_index_chexpert = cxr_labels_all_chexpert.index(disease_chexpert)
    disease_index_padchest = cxr_labels_all_padchest.index(disease_padchest)

    # Get for only the class "pleural effusion"
    pleural_effusion_chexpert = test_pred_chexpert[:,disease_index_chexpert]

    pleural_effusion_padchest = test_pred_padchest[:,disease_index_padchest]

    # Load in all true labels, filter for only the class "pleural effusion"
    cxr_labels_chexpert = [disease_chexpert]
    test_true_chexpert = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path_chexpert, cxr_labels=cxr_labels_chexpert)

    cxr_labels_padchest = [disease_padchest]
    test_true_padchest = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path_padchest, cxr_labels=cxr_labels_padchest)

    #######################
    # CHEXPERT
    # Sort probabilities in terms of their probabilities, keep the indices. 
    pleural_effusion_dict= dict(enumerate(pleural_effusion_chexpert))
    pleural_effusion_sorted = sorted(pleural_effusion_dict.items(), key=lambda pleural_effusion_dict: pleural_effusion_dict[1])

    auc_array_chexpert = []

    # Evaluate accuracy before removing anything. 

    bootstrap_results = eval.bootstrap(pleural_effusion_chexpert.reshape(-1, 1), test_true_chexpert, cxr_labels_chexpert, n_samples = 20) 
    x = bootstrap_results[0].at[0, disease_chexpert + '_auc']
    for ii in range(20):
        auc_array_chexpert.append(x.iloc[ii])


    middle_index = total_samples // 2
    i=2 
    while i < 0.8 * total_samples:
        # remove the middle element from "pleural_effusion_sorted". remove the same element from "test_true". Calculate the AUC for the non-removed part. 
        remove_indices = [x[0] for x in pleural_effusion_sorted[(middle_index-i//2):middle_index+i//2]]

        new_test_pred = np.delete(pleural_effusion_chexpert, remove_indices)

        new_test_true = np.delete(test_true_chexpert, remove_indices)

        bootstrap_results = eval.bootstrap(new_test_pred.reshape(-1, 1), new_test_true.reshape(-1,1), cxr_labels_chexpert, n_samples = 20) 

        x = bootstrap_results[0].at[0, disease_chexpert + '_auc']

        for ii in range(20):
            auc_array_chexpert.append(x.iloc[ii])

        i+=2
        

    x_axis_chexpert = x_axis_chexpert.repeat(20)

    sns.lineplot(x = x_axis_chexpert, y= auc_array_chexpert, label='CheXPert')

    #######################
    # Padchest
    # # Sort probabilities in terms of their probabilities, keep the indices. 
    # pleural_effusion_dict= dict(enumerate(pleural_effusion_padchest))
    # pleural_effusion_sorted = sorted(pleural_effusion_dict.items(), key=lambda pleural_effusion_dict: pleural_effusion_dict[1])

    # auc_array_padchest = []

    # # Evaluate accuracy before removing anything. 
    # bootstrap_results = eval.bootstrap(pleural_effusion_padchest.reshape(-1, 1), test_true_padchest, cxr_labels_padchest, n_samples = 20) 
    # x = bootstrap_results[0].at[0, disease_padchest + '_auc']

    # for ii in range(20):
    #     auc_array_padchest.append(x.iloc[ii])

    # middle_index = len(pleural_effusion_sorted) // 2
    # i=10 
    # while i < 0.8 * len(pleural_effusion_sorted):
    #     # remove the middle element from "pleural_effusion_sorted". remove the same element from "test_true". Calculate the AUC for the non-removed part. 
    #     remove_indices = [x[0] for x in pleural_effusion_sorted[(middle_index-i//2):middle_index+i//2]]

    #     new_test_pred = np.delete(pleural_effusion_padchest, remove_indices)

    #     new_test_true = np.delete(test_true_padchest, remove_indices)
        
    #     bootstrap_results = eval.bootstrap(new_test_pred.reshape(-1, 1), new_test_true.reshape(-1,1), cxr_labels_padchest,  n_samples = 20) 
    #     x = bootstrap_results[0].at[0, disease_padchest + '_auc']

    #     for ii in range(20):
    #         auc_array_padchest.append(x.iloc[ii])

    #     i+=10

    # x_axis = np.arange(0,int(0.8 * len(pleural_effusion_sorted)),10) / len(pleural_effusion_sorted) * 100 
    # x_axis = x_axis.repeat(20)
    
    # sns.lineplot(x = x_axis, y=auc_array_padchest, label='Padchest')

    plt.xlabel("Percentage of filtered out samples")
    plt.ylabel("AUC")
    plt.title('Edema')
    plt.savefig("Edema.png")
        
 