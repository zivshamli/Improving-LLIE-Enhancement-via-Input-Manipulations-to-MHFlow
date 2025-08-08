# Improving-LLIE-Enhancement-via-Input-Manipulations-to-MHFlow


## Introduction

Low-light image enhancement (LLIE) remains a challenging task due to issues such as noise, detail loss, and uneven illumination. In this work, we propose a modular approach to improve enhancement performance by injecting structural priors into the **input** of MHFlow—a normalizing flow-based model—**without modifying the model architecture**.

We investigate the effect of four types of input priors:
- **Unbalanced Point Map**
- **Illumination Map**
- **Edge Map**
- **Combined Prior (all three)**

Our experiments on the LOL and LOL-v2 datasets show that while global metrics (PSNR, SSIM) remain on par with the baseline, the proposed priors consistently improve performance in challenging regions, as reflected by the **LRC-PSNR** metric.

These findings highlight the potential of simple input-level guidance to improve model focus and suggest promising directions for future work:
- Learning priors
- Multi-prior integration
- Unpaired training data

---


## Datasets in our method

- LOL : [https://drive.google.com/drive/folders/1noCsfOZdCq14jp8FGdJ4-TqCgapLj35f?usp=sharing](https://drive.google.com/drive/folders/1noCsfOZdCq14jp8FGdJ4-TqCgapLj35f?usp=sharing)

- LOLv2 (Real & Synthetic): [https://drive.google.com/drive/folders/1noCsfOZdCq14jp8FGdJ4-TqCgapLj35f?usp=sharing](https://drive.google.com/drive/folders/1noCsfOZdCq14jp8FGdJ4-TqCgapLj35f?usp=sharing)



## Training our method

### Configuration

Modify the related parameters (paths, loss weights, training steps, and etc.) in the config yaml files
```bash
./conf/MHFlow.yml
```
### Train MHFlow

```bash
python train.py --opt config yaml file path
```

## Testing our method

### Pre-trained Models

Download the pre-trained models here:  
[https://drive.google.com/drive/folders/1u63uYdbadqQUzH3WBLKxeVfrhpG_2iNv?usp=sharing](https://drive.google.com/drive/folders/1u63uYdbadqQUzH3WBLKxeVfrhpG_2iNv?usp=sharing)

> **Note:**
> 1. First, choose the folder of the model that corresponds to the desired prior (Unbalanced Point Map, Illumination Map, Edge Map, or all maps).  
> 2. Inside that folder, select the model that was trained on the dataset you want to evaluate with.



### Run the testing code 

You can test the model with paired data and obtain the evaluation metrics. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
python test.py
```


## Contact
If you have any questions, please feel free to contact the authors via [zivshamli100@gmail.com](mailto:zivshamli100@gmail.com).
