# Improving-LLIE-Enhancement-via-Input-Manipulations-to-MHFlow


## Introduction
Our proposed method multiscale hybrid feature guided normalizing flow (MHFlow) is a novel and powerful generative model for low-light image enhancement. MHFlow can be trained using only NLL loss based on the estimation of the distribution. Extensive experiments on representative datasets show the superior performance of our method compared with current SOTA methods


## Datasets in our method

- LOLv2 (Real & Synthetic): Please refer to the papaer [[From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light).

- MIT: Please refer to the papaer [[Learning Enriched Features for Real Image Restoration and Enhancement (ECCV 2020)]](https://github.com/swz30/MIRNet).

- SMID: Please refer to the paper [[SNR-aware Low-Light Image Enhancement (CVPR 2022)]](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).

- Unpaired datasets (DICM, LIME, MEF, NPE and VV) [[Google Drive]](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T).

## Training our method

### Configuration

Modify the related parameters (paths, loss weights, training steps, and etc.) in the config yaml files
```bash
./conf/MHFlow.yml
```
### Train MHFlow

```bash
python train.py --opt config path
```

## Testing our method

### Pre-trained Models

Please download our pre-trained models via the following links [[Baiduyun (extracted code: og8u)]](https://pan.baidu.com/s/1MRnlYSQNSc5ZjxtfvAU5Vw?pwd=og8u) [[Google Drive]](https://drive.google.com/drive/folders/1Rax5fKq9QQOTcw9ancNQ75DeKdZKuVya?usp=sharing).

### Run the testing code 

You can test the model with paired data and obtain the evaluation metrics. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
python test.py
```


## Contact
If you have any questions, please feel free to contact the authors via [zivshamli100@gmail.com](zivshamli100@gmail.com).
