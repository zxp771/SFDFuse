# SFDFuse
Codes for ***SFDFuse: SFDFuse:Spatial and Frequency Feature Decomposition for Multi-Modality Image Fusion.***

[Xupei Zhang](https:///)
-[*[Paper]*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.html)  
-[*[ArXiv]*](https://arxiv.org/abs/2104.06977)  
-[*[Supplementary Materials]*](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_CVPR_2023_supplemental.pdf)  


## Update
- [2025/1] Release inference code for infrared-visible image fusion, Training codes and config files are public available.

## Citation

```

```

## Abstract

Multi-modal image fusion (MMIF) integrates information from multiple imaging modalities to generate high-quality fused images, with applications spanning multiple fields. Recent advances in deep learning have led to Deep Feature Decomposition (DFD) methods that leverage autoencoder networks to extract and fuse hierarchical spatial features, which obtained more comprehensive and richer visual information to
enhance the image fusion quality. However, these spatial-domain-focused methods remain limited in their ability to preserve essential details in regions of high grayscale variation and to balance global and local feature representations effectively. To overcome these challenges, this paper introduces a novel Spatial-Frequency feature Decomposition and Fusion (SFDFuse) network for MMIF, which incorporates both spatial and frequency domain features to enhance fusion quality by modified the spatial feature decompose modules and introduced the frequency feature decompose module. Moreover, this paper designed different feature fusion strategies to provide more efficient feature information complementation and information consistency representation for the cross domain and cross scale features. Extensive experiments demonstrate that SFDFuse achieves superior fusion performance, offering clearer scene representation and improved the detail preservation for the downstream vision tasks in a unified benchmark.

## 🌐 Usage

### ⚙ Network Architecture

Our SFDFuse is implemented in ``net.py``.

### 🏊 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate SFDfuse
# select pytorch version yourself
# install SFDfuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./MSRS_train/'``.

**3. Pre-Processing**

Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``.

**4. SFDFuse Training**

Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.

### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``'./models/SFDFuse_IVF.pth'`` , which is responsible for the Infrared-Visible Fusion (IVF). 

**2. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/RoadScene'``, ``'./test_img/TNO'`` for IVF, ``'./test_img/MRI_CT'``, ``'./test_img/MRI_PET'`` and ``'./test_img/MRI_SPECT'`` for MIF.

Unfortunately, since the size of **MSRS dataset** for IVF is 500+MB, we can not upload it for exhibition. It can be downloaded via [this link](https://github.com/Linfeng-Tang/MSRS). The other datasets contain all the test images.

**3. Results in Our Paper**

If you want to infer with our CDDFuse and obtain the fusion results in our paper, please run 
```
python test.py
``` 
for Infrared-Visible Fusion

The testing results will be printed in the terminal. 

The output for ``'test_IVF.py'`` is:

```
================================================================================
The test result of TNO :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
CDDFuse         7.12    46.0    13.15   2.19    1.76    0.77    0.54    1.03
================================================================================

================================================================================
The test result of RoadScene :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
CDDFuse         7.44    54.67   16.36   2.3     1.81    0.69    0.52    0.98
================================================================================
```
which can match the results in Table 1 in our original paper.


## 🙌 CDDFuse

### Illustration of our SFDFuse model.

<img src="image//Workflow.png" width="90%" align=center />

### Qualitative fusion results.

<img src="image//IVF1.png" width="90%" align=center />

<img src="image//IVF2.png" width="90%" align=center />

<img src="image//MIF.png" width="60%" align=center />

### Quantitative fusion results.

Infrared-Visible Image Fusion

<img src="image//Quantitative_IVF.png" width="60%" align=center />

Medical Image Fusion

<img src="image//Quantitative_MIF.png" width="60%" align=center />

MM detection

<img src="image//MMDet.png" width="60%" align=center />

MM segmentation

<img src="image//MMSeg.png" width="60%" align=center />


## 📖 Related Work

- Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, Luc Van Gool. *Equivariant Multi-Modality Image Fusion.* **arXiv:2305.11443**, https://arxiv.org/abs/2305.11443

- Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, Luc Van Gool.
*DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion.* **ICCV 2023**, https://arxiv.org/abs/2303.06840

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang and Pengfei Li. *DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion.* **IJCAI 2020**, https://www.ijcai.org/Proceedings/2020/135.

- Zixiang Zhao, Shuang Xu, Jiangshe Zhang, Chengyang Liang, Chunxia Zhang and Junmin Liu. *Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling.* **IEEE Transactions on Circuits and Systems for Video Technology 2021**, https://ieeexplore.ieee.org/document/9416456.

- Zixiang Zhao, Jiangshe Zhang, Haowen Bai, Yicheng Wang, Yukun Cui, Lilun Deng, Kai Sun, Chunxia Zhang, Junmin Liu, Shuang Xu. *Deep Convolutional Sparse Coding Networks for Interpretable Image Fusion.* **CVPR Workshop 2023**. https://robustart.github.io/long_paper/26.pdf.

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang. *Bayesian fusion for infrared and visible images.* **Signal Processing**, https://doi.org/10.1016/j.sigpro.2020.107734.

