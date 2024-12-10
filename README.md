<div align="center">
<h1> E2E-MFD </h1>
<h3> <a href="https://arxiv.org/abs/2403.09323">E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection</h3>
<h4> NeurlPS 2024 oral</h4>
  
</div>

The code is based on **[MMdetection](https://github.com/open-mmlab/mmdetection) 2.26.0**, **[MMrotate](https://github.com/open-mmlab/mmrotate/) 0.3.4** and **[MMCV-full](https://github.com/open-mmlab/mmcv) 1.7.2**. We modify its data loading, related classes, and functions. We revise the MMdetection and MMrotate to a multi-modal oriented detection framework to facilitate **Multimodal Object Detection**.


## **Overview**

<p align="center">
  <img src="assets\frame.png" alt="overview" width="90%">
</p>

## **Getting Started**

### Installation

ref : [mmrotate installation](https://mmrotate.readthedocs.io/en/latest/install.html#installation) and [mmdetection installation](https://mmdetection.readthedocs.io/en/latest/get_started.html)

**Step 1: Clone the E2E-MFD repository:**

To get started, first clone the E2E-MFD repository and navigate to the project directory:

```bash
git clone https://github.com/icey-zhang/E2E-MFD
cd E2E-MFD
```

**Step 2: Environment Setup:**

E2E-MFD recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n E2E-MFD python=3.9.17
conda activate E2E-MFD
```

***If you develop and run mmrotate directly, install it from source***

```bash
pip install -v -e .
```

***Install Dependencies***

```bash
pip install -r requirements.txt
```

### Prepare the dataset DroneVehicle
[DroneVehicle]((https://ieeexplore.ieee.org/abstract/document/9759286)) is a publicly available dataset. 

you can download the dataset at baiduyun with [train](https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg) (code:ngar) and [test](https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg) (code:tqwc).

```python
root
├── DroneVehicle
│   ├── train
│   │   ├── rgb
│   │   │   ├── images
│   │   │   ├── labels
│   │   ├── ir
│   │   │   ├── images
│   │   │   ├── labels
│   ├── test
│   │   ├── rgb
│   │   │   ├── images
│   │   │   ├── labels
│   │   ├── ir
│   │   │   ├── images
│   │   │   ├── labels
```

### Begin to train and test

Use the config file with [this](./tools/cfg/lsk_s_fpn_1x_dota_le90.py).

```python
python ./tools/train.py
python ./tools/test.py
```

### Generate fusion images

```python
python ./tools/generate_fusion_image.py
```

## Result

[DroneVehicle weights](https://drive.google.com/file/d/1U_u1s61sb0-SrkcUEb3HaDVLZtznCRV7/view?usp=sharing) <br>
[DroneVehicle logs](./assets/train.log)

## Citation
If our code is helpful to you, please cite:

```
@ARTICLE{10075555,
  author={Zhang, Jiaqing and Lei, Jie and Xie, Weiying and Fang, Zhenman and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SuperYOLO: Super Resolution Assisted Object Detection in Multimodal Remote Sensing Imagery}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3258666}}

@article{zhang2023guided,
  title={Guided Hybrid Quantization for Object Detection in Remote Sensing Imagery via One-to-one Self-teaching},
  author={Zhang, Jiaqing and Lei, Jie and Xie, Weiying and Li, Yunsong and Yang, Geng and Jia, Xiuping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}

@inproceedings{zhange2e,
  title={E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection},
  author={Zhang, Jiaqing and Cao, Mingxiang and Xie, Weiying and Lei, Jie and Huang, Wenbo and Li, Yunsong and Yang, Xue},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}


```
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=icey-zhang/E2E-MFD&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=icey-zhang/E2E-MFD&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=icey-zhang/E2E-MFD&type=Date"
    width="600"  <!-- 设置宽度 -->
    height="500" <!-- 可选设置高度 -->
  />
</picture>



