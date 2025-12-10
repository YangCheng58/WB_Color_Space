# Perception-Inspired Color Space Design for Photo White Balance Editing

> **Official PyTorch Implementation** of our WACV 2026 paper.

**[Yang Cheng](https://github.com/YangCheng58/)**<sup>1</sup>, **[Ziteng Cui](https://cuiziteng.github.io/)**<sup>2, *</sup>, **[Lin Gu](https://sites.google.com/view/linguedu/home/)**<sup>3</sup>, **[Shenghan Su](https://github.com/ryeocthiv/)**<sup>1</sup>, **Zenghui Zhang**<sup>1</sup>

<small>
<sup>1</sup> Shanghai Jiao Tong University <br>
<sup>2</sup> The University of Tokyo <br>
<sup>3</sup> Tohoku University
</small>

<br>

<small>* Corresponding author</small>


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
git clone https://github.com/YangCheng58/WB_Color_Space.git
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
```

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
  
## 📈 Evaluation

We provide a comprehensive evaluation script to test the model on **Set1**, **Set2**, and **Cube+** datasets. The script calculates **MSE**, **MAE**, and **$\Delta E_{00}$**, reporting both the Mean and Quartiles (Q1, Median, Q3).

### 1. Pre-trained Model
The pre-trained model weights are already provided in this repository at:
> `models/best.pth`

### 2. Run Evaluation
You can evaluate specific datasets using the following commands.

**Evaluate on Set1:**
```bash
python eval.py \
  --dataset Set1 \
  --data_root ./dataset/Set1_all \
  --split_file ./folds/fold3_.mat \
  --model_path models/best.pth
```

**Evaluate on Set2:**
```bash
python eval.py \
  --dataset Set2 \
  --input_dir ./dataset/Set2_input_images \
  --gt_dir ./dataset/Set2_ground_truth_images \
  --model_path models/best.pth
```

**Evaluate on Cube+:**
```bash
python eval.py \
  --dataset Cube \
  --input_dir ./dataset/Cube_input_images \
  --gt_dir ./dataset/Cube_ground_truth_images \
  --model_path models/best.pth
```

### 3. Output Metrics
The script will output a table containing Mean, Q1 (25%), Median (50%), and Q3 (75%) for all metrics. It also automatically saves any outliers (MSE > 500) to a text file for further analysis.


