# Perception-Inspired Color Space Design for Photo White Balance Editing

> **Official PyTorch Implementation** of our WACV 2026 paper.


### 🚀 Framework Overview

#### 1. The Learnable HSI (LHSI) Color Space
Our proposed color space introduces a learnable luminance axis and adaptive nonlinear mapping functions.

<div align="center">
  <img src="pics/colorspace.png" width="800"/>
</div>

#### 2. DCLAN Architecture
The overall pipeline utilizes the **MambaVision Module (MVM)** and **Cross Attention Module (CAM)** to effectively process the disentangled features.

<div align="center">
  <img src="pics/overallnetwork.png" width="100%"/>
</div>

## ⚡ Get Started


* **System**: Linux (Recommended for Mamba compilation)
* **Python**: 3.10 (Recommended)
* **CUDA**: 11.8 (Required for Mamba-SSM)
* **PyTorch**: 2.0+

```bash
git clone [https://github.com/YangCheng58/WB_Color_Space.git](https://github.com/YangCheng58/WB_Color_Space.git)
```

```bash
cd WB_Color_Space
```

```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation

We evaluate our method using the **Rendered WB Dataset** introduced by Afifi et al. in [Correcting Improperly White-Balanced Images (CVPR 2019)](https://github.com/mahmoudnafifi/WB_sRGB).

Following the data organization and evaluation protocol described in [Deep White-Balance Editing (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Afifi_Deep_White-Balance_Editing_CVPR_2020_paper.pdf), please download the dataset and organize the files as follows.

### 1. Download Dataset
Please download the full dataset (Set1, Set2, and Cube+) from the **[Official Dataset Repository](https://github.com/mahmoudnafifi/WB_sRGB)**.

### 2. Directory Structure
Extract and organize the data into the `./dataset/` directory:

```text
./dataset/
├── Set1_all/                  # Contains Inputs and GTs from Set1
├── Set2_input_images/         # Testing inputs from Set2
├── Set2_ground_truth_images/  # Testing GTs from Set2
├── Cube_input_images/         # Testing inputs from Cube+
└── Cube_ground_truth_images/  # Testing GTs from Cube+

**Note**: We follow the standard cross-validation protocol. Specifically, we use Fold 3 as the testing set. The detailed image list and split definition can be found in folds/fold3_.mat.


## 🚀 Training

To train the DCLAN model from scratch on Set1 using the standard Fold 3 configuration, run the following command:

```bash
python train.py \
  --training_dir ./dataset/Set1_all \
  --fold 3 \
  --epochs 120 \
  --num_training_images 12000
```
  
📊 Evaluation

We provide a comprehensive evaluation script eval.py to test on Set1, Set2, and Cube+ datasets.

Change the datasets and settings in eval.py and run it

