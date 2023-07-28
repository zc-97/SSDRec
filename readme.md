# SSDRec: Self-Augmented Sequence Denoising for Sequential Recommendation

## Requirements
***
Our model SSDRec is implemented based on the RecBole v1.0.1. Both the processing of the dataset and the metrics calculation follow the implementation of RecBole.
* python 3.70+
* PyTorch 1.7.1+
* yaml 6.0+
* openpyxl 3.0.9+
* RecBole 1.0.1+
* tqdm 4.64.0
  
Specifically, we implement the denoising function in the third stage of SSDRec by integrating with [HSD](https://github.com/zc-97/HSD)

## Preparing Environment
***
### Install [Recbole](https://github.com/RUCAIBox/RecBole), 
#### Install from Conda
```commandline
conda install -c aibox recbole
```
#### Install from pip
```commandline
pip install recbole
```
#### Install from source
```commandline
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```
### Usage
