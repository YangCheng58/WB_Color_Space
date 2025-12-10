"""
 Dataloader
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import os
from os.path import join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance 

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, fold=0, patch_size=128, patch_num_per_image=1, max_trdata=12000):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_image = patch_num_per_image
        
        if fold != 0:
            tfolds = list(set([1, 2, 3]) - set([fold]))
            logging.info(f'Training process will use {max_trdata} training images randomly selected from folds {tfolds}')
            
            files = loadmat(join('folds', 'fold%d_.mat' % fold))
            
            files = files['training']
            self.imgfiles = []
            logging.info('Loading training images information...')
            for i in range(len(files)):
               
                temp_files = glob(join(imgs_dir, files[i][0][0]))
                for file in temp_files:
                    self.imgfiles.append(file)
        elif fold == 0:
            logging.info(f'Training process will use {max_trdata} training images randomly selected from all training data')
            logging.info('Loading training images information...')
            self.imgfiles = [join(imgs_dir, file) for file in listdir(imgs_dir)
                             if not file.startswith('.')]
        else:
            logging.info(f'There is no fold {fold}! Training process will use all training data.')
        
        if max_trdata != 0 and len(self.imgfiles) > max_trdata:
            random.shuffle(self.imgfiles)
            self.imgfiles = self.imgfiles[0:max_trdata]
        logging.info(f'Creating dataset with {len(self.imgfiles)} examples')

    def __len__(self):
        return len(self.imgfiles)

    @staticmethod
    def randomCT(image):
        random_factor = np.random.randint(0, 31) / 10. 
        enhancer = ImageEnhance.Color(image)
        color_image = enhancer.enhance(random_factor)
        return color_image

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op, apply_ct=False):
        patch_x, patch_y = patch_coords
        pil_patch = pil_img.crop((patch_x, patch_y, patch_x + patch_size, patch_y + patch_size))

        if flip_op == 1:
            pil_patch = ImageOps.mirror(pil_patch)
        elif flip_op == 2:
            pil_patch = ImageOps.flip(pil_patch)

        if apply_ct:
            pil_patch = cls.randomCT(pil_patch)

        img_nd = np.array(pil_patch)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255.0

        return img_trans

    def __getitem__(self, i):
        gt_ext = 'G_AS.png'
        img_file = self.imgfiles[i]
        in_img = Image.open(img_file)
        w, h = in_img.size

        parts = img_file.split('_')
        base_name = ''
        for k in range(len(parts) - 2):
            base_name = base_name + parts[k] + '_'
        gt_awb_file = base_name + gt_ext
        awb_img = Image.open(gt_awb_file)

        flip_op = np.random.randint(3)
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        
        ct_trigger = np.random.randint(3) 
        do_augment = (ct_trigger != 0) 

        in_img_patches = self.preprocess(
            in_img, self.patch_size, (patch_x, patch_y), flip_op, apply_ct=do_augment
        )
        
        awb_img_patches = self.preprocess(
            awb_img, self.patch_size, (patch_x, patch_y), flip_op, apply_ct=False
        )

        for j in range(self.patch_num_per_image - 1):
            flip_op = np.random.randint(3)
            patch_x = np.random.randint(0, high=w - self.patch_size)
            patch_y = np.random.randint(0, high=h - self.patch_size)

            ct_trigger = np.random.randint(2)
            do_augment = (ct_trigger != 0)

            temp_in = self.preprocess(
                in_img, self.patch_size, (patch_x, patch_y), flip_op, apply_ct=do_augment
            )
            in_img_patches = np.append(in_img_patches, temp_in, axis=0)
            
            temp_awb = self.preprocess(
                awb_img, self.patch_size, (patch_x, patch_y), flip_op, apply_ct=False
            )
            awb_img_patches = np.append(awb_img_patches, temp_awb, axis=0)

        return {'image': torch.from_numpy(in_img_patches), 'gt-AWB': torch.from_numpy(awb_img_patches)}