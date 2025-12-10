Perception-Inspired Color Space Design for Photo White Balance Editing

[WACV 2026 Accepted] Official PyTorch Implementation.

📖 Introduction

Abstract: White balance (WB) is a key step in the image signal processor (ISP) pipeline that mitigates color casts caused by varying illumination and restores the scene’s true colors. Currently, sRGB-based WB editing for post-ISP WB correction is widely used to address color constancy failures in the ISP pipeline when the original camera RAW is unavailable. However, additive color models (e.g., sRGB) are inherently limited by fixed nonlinear transformations and entangled color channels, which often impede their generalization to complex lighting conditions.

To address these challenges, we propose a novel framework for WB correction that leverages a perception-inspired Learnable HSI (LHSI) color space. Built upon a cylindrical color model that naturally separates luminance from chromatic components, our framework further introduces dedicated parameters to enhance this disentanglement and learnable mapping to adaptively refine the flexibility. Moreover, a new Mamba-based network is introduced, which is tailored to the characteristics of the proposed LHSI color space. 

Experimental results on benchmark datasets demonstrate the superiority of our method, highlighting the potential of perception-inspired color space design in computational photography.

![Overview of the Learnable HSI (LHSI) color space.](pics/colorspace.png)

![Overview of our DCLAN architecture.](pics/.png)


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

