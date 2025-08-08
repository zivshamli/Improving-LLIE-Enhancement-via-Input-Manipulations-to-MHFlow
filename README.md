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


## Hardware Requirements

For optimal performance, we recommend running the training and testing on a GPU with:
- **At least 8 GB of VRAM**
- **CUDA support** enabled

Running on CPU is possible but will be significantly slower.


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
---
## References

- Hu, C., Hu, Y., Xu, L., Cai, Z., Wu, F., Jing, X., & Lu, X. (2025). Multiscale hybrid feature guided normalizing flow for low-light image enhancement. *Computers and Electrical Engineering*, 122, 109922. [https://www.sciencedirect.com/science/article/abs/pii/S0045790624008486](https://www.sciencedirect.com/science/article/abs/pii/S0045790624008486)

- Xu, L., Hu, C., Hu, Y., Jing, X., Cai, Z., & Lu, X. (2025). UPT-Flow: Multi-scale transformer-guided normalizing flow for low-light image enhancement. *Pattern Recognition*, 158, 111076. [https://www.sciencedirect.com/science/article/abs/pii/S0031320324008276?casa_token=IrqVBXpTT_YAAAAA:Jr_NQTVyyklmfDzNnwoNRirsMg2kjYg_8f1rgXZC6XGXukL-YSIWHb0trBRnPWqV9z7sJBaV5hA](https://www.sciencedirect.com/science/article/abs/pii/S0031320324008276?casa_token=IrqVBXpTT_YAAAAA:Jr_NQTVyyklmfDzNnwoNRirsMg2kjYg_8f1rgXZC6XGXukL-YSIWHb0trBRnPWqV9z7sJBaV5hA)

- Guo, X., Li, Y., & Ling, H. (2016). LIME: Low-light image enhancement via illumination map estimation. *IEEE Transactions on Image Processing*, 26(2), 982–993. [https://ieeexplore.ieee.org/document/7782813](https://ieeexplore.ieee.org/document/7782813)




## Contact
If you have any questions, please feel free to contact the authors via [zivshamli100@gmail.com](mailto:zivshamli100@gmail.com).
