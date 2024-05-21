<div align="center">
<h1> E2E-MFD </h1>
<h3> E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection</h3>

</div>

The code is based on **[MMdetection](https://github.com/open-mmlab/mmdetection) 2.26.0**, **[MMrotate](https://github.com/open-mmlab/mmrotate/) 0.3.4** and **[MMCV-full](https://github.com/open-mmlab/mmcv) 1.7.2**. We modify its data loading, related classes, and functions. We revise the MMdetection and MMrotate to a multi-modal oriented detection framework to facilitate **Multimodal Object Detection**.

## **Overview**

<p align="center">
  <img src="assets\frame.png" alt="overview" width="90%">
</p>

## **Getting Started**

### Installation

ref : [mmdetection installation](https://mmdetection.readthedocs.io/en/latest/get_started.html)

**Step 1: Clone the RSDet repository:**

To get started, first clone the RSDet repository and navigate to the project directory:

```bash
git clone https://github.com/Zhao-Tian-yi/RSDet.git
cd RSDet
```

**Step 2: Environment Setup:**

RSDet recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n RSDet
conda activate RSDet
```

***If you develop and run mmdet directly, install it from source***

```
pip install -v -e .
```

***Install Dependencies***

```bash
pip install -r requirements.txt
pip install -r requirements_rgbt.txt
```

## **Result**

[Kaist Result](https://drive.google.com/file/d/11tiHFCRG8ubt23g-BN1wY3W94pYNimL0/view?usp=sharing)

## **Future**
The paper is under review, and this code repository will be updated after accepted
