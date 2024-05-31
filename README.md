# Calibration and Uncertainty Estimation Challenges in Self-Supervised Chest X-ray Pathology Classification Models
Jenny Xu, Pranav Rajpurkar. \
2024 Medical Imaging with Deep Learning, Paris, France. \
arxiv link to be added. 

This repository is based off [CheXzero](https://github.com/rajpurkarlab/CheXzero).  

## Overview
- [Abstract](#abstract)
- [Changelog](#changelog)
- [Installation](#installation)
- [Zero-Shot Inference](#zero-shot-inference)
- [Uncertainty Estimation](#uncertainty-estimation)
- [License](#license)
- [Issues](#issues)

## Abstract
Uncertainty quantification is crucial for the safe deployment of AI systems in clinical radiology. We analyze the calibration of [CheXzero](https://github.com/rajpurkarlab/CheXzero), a high-performance self-supervised model for chest X-ray pathology detection, on two external datasets and evaluate the effectiveness of two common uncertainty estimation methods: Maximum Softmax Probabilities (MSP) and Monte Carlo Dropout. Our analysis reveals poor calibration on both external datasets, with Expected Calibration Error (ECE) scores ranging from 0.12 to 0.41. Furthermore, we find that the model's prediction accuracy does not correlate with the uncertainty scores derived from MSP and Monte Carlo Dropout. These findings highlight the need for more robust uncertainty quantification methods to ensure the trustworthiness of AI-assisted clinical decision-making. 

## Changelog
[2024-05-30] v0.1.0 is released. 

## Installation
Please follow [CheXzero](https://github.com/rajpurkarlab/CheXzero) for environment setup and model checkpoints. 

## Zero-Shot Inference
1. Follow the section "Zero-Shot Inference" of [CheXzero](https://github.com/rajpurkarlab/CheXzero) and use the file [`notebooks/zero_shot_chexpert.py`](notebooks/zero_shot_chexpert.py) to perform zero-shot inference on CheXpert dataset. The output predictions are obtained from the model ensemble. 
2. Use the file [`notebooks/zero_shot_padchest.py`](notebooks/zero_shot_padchest.py) to perform zero-shot inference on PadChest dataset. The output predictions are obtained from the model ensemble. 
3. Use the file [`notebooks/zero_shot_mc_chexpert.py`](notebooks/zero_shot_mc_chexpert.py) to perform zero-shot inference on CheXpert dataset using Monte Carlo Dropout. A set of 30 output predictions are generated using 30 runs of Monte Carlo Dropout. 
4. Use the file [`notebooks/zero_shot_mc_padchest.py`](notebooks/zero_shot_mc_padchest.py) to perform zero-shot inference on PadChest dataset using Monte Carlo Dropout. A set of 30 output predictions are generated using 30 runs of Monte Carlo Dropout. 

## Uncertainty Estimation
1. Run the following command to plot the results of uncertainty estimation using the three techniques: Median Removal, MSP, MCD, on the CheXpert dataset. 
```bash
cd plot_scripts
python run_chexpert_UQ.py --condition 'Pleural Effusion' --gt_path  'data/groundtruth.csv' --pred_path 'predictions/ensemble/chexpert_preds.npy' --MCD_pred_path 'predictions/chexpert_MCD/'
```

### Arguments
* `--condition` one of ['Pleural Effusion', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema']
* `--gt_path` The path to the ground truth labels csv file.
* `--pred_path` The path to the predictions.
* `--MCD_pred_path` The path to the MCD predictions.

Use `-h` flag to see all optional arguments. 

2. Run the following command to plot the results of uncertainty estimation on PadChest dataset. 
```bash
cd plot_scripts
python run_padchest_UQ.py --condition 'pleural effusion' --gt_path  'data/padchest/2_cxr_labels.csv' --pred_path 'predictions/padchest_ensemble/padchest_preds.npy' --MCD_pred_path 'predictions/padchest_MCD/'
```

### Arguments
* `--condition` one of ['pleural effusion', 'atelectasis', 'cardiomegaly', 'consolidation', 'pulmonary edema']
* `--gt_path` The path to the ground truth labels csv file.
* `--pred_path` The path to the predictions.
* `--MCD_pred_path` The path to the MCD predictions.

## License 
The source code for the site is licensed under the MIT license, which you can find in the `LICENSE` file. The source code is based off [CheXzero](https://github.com/rajpurkarlab/CheXzero). Also see `NOTICE.md` for the changelog. 

## Issues 
Please open new issue threads specifying the issue with the codebase or report issues directly to [jennyxu6@stanford.edu](jennyxu6@stanford.edu).

