Perception-Inspired Color Space Design for Photo White Balance Correction

[WACV 2026 Accepted] Official PyTorch Implementation.

📖 Introduction

Abstract: White Balance (WB) is a critical component of the Image Signal Processor (ISP) pipeline, designed to mitigate color casts introduced by diverse illumination conditions and restore the scene's true colors. Currently, sRGB-based methods are widely adopted for post-ISP WB editing where the original raw data is unavailable. However, this task is ill-posed due to the non-linearity introduced by the ISP and the entanglement of color channels in the sRGB domain.

To address these challenges, we propose a novel framework leveraging a Learnable HSI (LHSI) color space. By disentangling luminance from chromatic components via our proposed DCLAN (Deep Color-Space-Aware Network), which integrates MambaVision blocks for efficient long-range dependency modeling, the LHSI representation facilitates more effective modeling of illumination changes.

Figure 1: Overview of the Learnable HSI (LHSI) color space.

Figure 2: Overview of our DCLAN architecture.

🛠️ Environment Requirements

This project relies on torch, timm, and mamba_ssm. Please ensure you have a GPU environment set up.

Recommended: Linux, CUDA 11.8+, Python 3.8+

📂 Dataset Preparation

We use the dataset from "Correcting Improperly White-Balanced Images" (CVPR 2019).
Please download the dataset from the official repository.

Directory Structure

Please organize the downloaded dataset as follows:

./dataset/
├── Set1_all/                # Contains Set1 images and GTs
├── Set2_input_images/       # Set2 Inputs
├── Set2_ground_truth_images/# Set2 GTs
├── Cube_input_images/       # Cube+ Inputs
├── Cube_ground_truth_images/# Cube+ GTs



Note: The folds/*.mat files define the train/test split for Set1, which are typically provided with the original dataset or can be found in our folds/ directory.

🚀 Training

To train the DCLAN model from scratch on Set1.

python train.py \
  --training_dir ./dataset/Set1_all \
  --fold 0 \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --patches-per-image 4 
  
📊 Evaluation

We provide a comprehensive evaluation script eval.py to test on Set1, Set2, and Cube+ datasets.

Change the datasets and settings in eval.py and run it

