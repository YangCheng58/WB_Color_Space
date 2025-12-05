"""
 Training
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from utilities.dataset import TrainDataset,ValDataset
from torch.utils.data import DataLoader
from utilities.loss_func import mae_loss
from arch.DCLAN import DCLAN
    
def train_net(net,
              device,
              epochs=110,
              batch_size=128,
              lr=0.0001,
              lrdf=0.5,
              fold=0,
              trimages=12000,
              patchsz=256,
              patchnum=4,
              dir_img='/workspace/users/cy/renderWB/Set1_all'):
    dir_checkpoint = f'/workspace/users/cy/checkpoints_{fold}_lhsi/'
    train = TrainDataset(dir_img, fold=fold, patch_size=patchsz, patch_num_per_image=patchnum, max_trdata=trimages)
    val=ValDataset()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(net.parameters(), 
                      lr=lr, 
                      weight_decay=0, 
                      betas=(0.9, 0.99))

    scheduler = MultiStepLR(optimizer, 
                        milestones=[40, 80, 120, 160, 200,240,260,270,280,290], 
                        gamma=lrdf)

    for epoch in range(epochs):
        net.train()
        epoch_loss = []

        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            loss_all = 0
            test_loss_all=0
            for batch in train_loader:
                imgs_ = batch['image']
                awb_gt_ = batch['gt-AWB']

                for j in range(patchnum):
                    imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                    awb_gt = awb_gt_[:, (j * 3): 3 + (j * 3), :, :]

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    awb_gt = awb_gt.to(device=device, dtype=torch.float32)

                    optimizer.zero_grad()

                    img_rgb,normal= net(imgs)
                    img_rgb = img_rgb.clamp(min=0, max=1)
                    awb_hsv=net.rgb2hsv(awb_gt)
                    img_hsv=net.rgb2hsv(img_rgb)
                    img_rgb = img_rgb.clamp(min=0, max=1)

                    loss_rgb=mae_loss.compute(img_rgb,awb_gt)  
                    loss_hvi=mae_loss.compute(img_hsv,awb_hsv)
                    total_loss = 0.9*loss_rgb + 0.1*loss_hvi
            
                    total_loss = total_loss.float()
                    total_loss.backward()

                    optimizer.step()

                    loss_all += total_loss.item()
                    test_loss_all+=mae_loss.compute(img_rgb,awb_gt).item()
        
                    pbar.set_postfix(**{'loss (batch)': total_loss.item()})
                    pbar.update(np.ceil(imgs.shape[0] / patchnum))

        scheduler.step()
        loss_all /= len(train_loader) * imgs.shape[0]
        epoch_loss.append(loss_all)
        test_loss_all/=len(train_loader)*imgs.shape[0]
        print(normal)
        print(loss_all)
        print(test_loss_all)

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')

        torch.save(net.state_dict(), dir_checkpoint + f'deep_WB_epoch{epoch +1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved!')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')

    torch.save(net.state_dict(), 'models/' + 'netlhsi.pth')
    print(epoch_loss)
    logging.info('End of training')


def get_args():
    parser = argparse.ArgumentParser(description='Train deep WB editing network.')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=110,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-vf', '--validation-frequency', dest='val_frq', type=int, default=2,
                        help='Validation frequency.')
    parser.add_argument('-d', '--fold', dest='fold', type=int, default=1,
                        help='Testing fold to be excluded. Use --fold 0 to use all Set1 training data')
    parser.add_argument('-p', '--patches-per-image', dest='patchnum', type=int, default=4,
                        help='Number of training patches per image')
    parser.add_argument('-s', '--patch-size', dest='patchsz', type=int, default=128,
                        help='Size of training patch')
    parser.add_argument('-t', '--num_training_images', dest='trimages', type=int, default=13333,
                        help='Number of training images. Use --num_training_images 0 to use all training images')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-ldf', '--learning-rate-drop-factor', dest='lrdf', type=float, default=0.5,
                        help='Learning rate drop factor')
    parser.add_argument('-ldp', '--learning-rate-drop-period', dest='lrdp', type=int, default=25,
                        help='Learning rate drop period')
    parser.add_argument('-trd', '--training_dir', dest='trdir', default='../dataset/',
                        help='Training image directory')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of Deep White-Balance Editing')

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = DCLAN()
    net.to(device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrdf=args.lrdf,
                  device=device,
                  fold=args.fold,
                  trimages=args.trimages,
                  patchsz=args.patchsz,
                  patchnum=args.patchnum,
                  dir_img=args.trdir)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
