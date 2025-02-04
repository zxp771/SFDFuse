# SFDFuse
Codes for ***SFDFuse: SFDFuse:Spatial and Frequency Feature Decomposition for Multi-Modality Image Fusion.***

[Xupei Zhang](https:///)
-[*[Paper]*]()  
-[*[ArXiv]*]()  
-[*[Supplementary Materials]*]()  


## Update
- [2025/1] Release inference code for infrared-visible image fusion, Training codes and config files are public available.


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
conda create -n SFDfuse python=3.8.10
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

Pretrained models are available in ``'./models/SFDFuse_best.pth'`` , which is responsible for the Infrared-Visible Fusion (IVF). 

**2. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/MSRS'``,``'./test_img/RoadScene'``, ``'./test_img/TNO'``,``'./test_img/M3FD'`` for IVF.

Unfortunately, since the size of train and test dataset is over 500+MB, we can not upload them for exhibition. It can be downloaded via the links in data/datasetlink.txt and test_data/datasetlinks.txt.

**3. Results in Our Paper**

If you want to infer with our CDDFuse and obtain the fusion results in our paper, please run 
```
python test.py
``` 
for Infrared-Visible Fusion

The testing results will be printed in the terminal. 

The output for ``'test.py'`` is:

```
================================================================================
The test result of MSRS :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
SFDFuse         6.70    43.40    11.59   3.54    1.65    1.06    0.71    1.00
================================================================================

================================================================================
The test result of TNO :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
SFDFuse         7.16    46.17    13.24   2.29    1.74    0.81    0.56    1.02
================================================================================

================================================================================
The test result of RoadScene :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
SFDFuse         7.43    52.80   15.63   2.37     1.80    0.70    0.55    0.96
================================================================================

================================================================================
The test result of TNO :
                 EN      SD      SF      MI     SCD     VIF     Qabf    SSIM
SFDFuse         6.93    38.02    14.96   2.98    1.59    0.83    0.63    1.01
================================================================================
```
which can match the results in Table 1 in our original paper.


## 🙌 SFDFuse

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

## Related Works

```
- Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, Luc Van Gool. *Equivariant Multi-Modality Image Fusion.* **arXiv:2305.11443**, https://arxiv.org/abs/2305.11443

- Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, Luc Van Gool.
*DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion.* **ICCV 2023**, https://arxiv.org/abs/2303.06840

- Zhao, Z., Bai, H., Zhang, J., Zhang, Y., Xu, S., Lin, Z., Timofte, R., Van Gool, L., 2023a. Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.5906–5916.

- Hu, X., Liu, Y., Yang, F., 2024. Pfcfuse: A poolformer and cnn fusion network for infrared-visible image fusion. IEEE Transactions on Instrumentation and Measurement.

- Li, H., Wu, X.J., 2024. Crossfuse: A novel cross attention mechanism based infrared and visible image fusion approach. Information Fusion 103,102147.

- Li, H., Wu, X.J., Kittler, J., 2021. Rfn-nest: An end-to-end residual fusion network for infrared and visible images. Information Fusion 73, 72–86.
```


## Citation

```

```



