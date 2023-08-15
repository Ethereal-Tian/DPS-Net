from __future__ import print_function, division
from genericpath import exists
import os
from signal import default_int_handler

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
from calendar import c
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime 
import time
import matplotlib.pyplot as plt
import cv2
from PIL import ImageEnhance, Image
# from sklearn import metrics

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from core.dps_net import DPSNet, autocast
print('core.dps_net')

from core.utils.utils import InputPadder
import core.rgbp_stereo_datasets as datasets
from core.utils import frame_utils

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def eval(args,model, eval_dataset):
    model.eval()
    torch.backends.cudnn.benchmark=True

    bad1_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    bad2_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    bad3_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    epe_tensor = torch.zeros(len(eval_dataset)).cuda().float()    
    elapsed_tensor = torch.zeros(len(eval_dataset)).cuda().float()    
    rmse_tensor = torch.zeros(len(eval_dataset)).cuda().float()    
    
    with torch.no_grad():
        for eval_id in tqdm(range(len(eval_dataset))):
            _, imgL, imgR, polL, polR, disp_gt, valid_gt = eval_dataset[eval_id]        
        
            imgL = imgL[None].cuda()
            imgR = imgR[None].cuda()
            polL = polL[None].cuda()
            polR = polR[None].cuda()
            disp_gt = disp_gt.cuda()
            valid_gt = valid_gt.cuda()

            padder = InputPadder(imgL.shape, divis_by=32)
            imgL, imgR, polL, polR = padder.pad(imgL, imgR, polL, polR)

            with autocast(enabled=args.mixed_precision):
                start = time.time()
                _, disp_pr = model.module(imgL, imgR, polL, polR, gru_iters=args.valid_gru_iters, test_mode=True)
                end = time.time()
            
            disp_pr = padder.unpad(disp_pr).squeeze(0)
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.sum((disp_pr - disp_gt)**2, dim=0).sqrt()                              

            epe_flattened = epe.flatten()
            val = (valid_gt.flatten() >= 0.5)  

            b1 = (epe_flattened > 1.0) #1px
            b2 = (epe_flattened > 2.0) #2px
            b3 = (epe_flattened > 3.0) #3px            
            image_epe = epe_flattened[val].mean().item()               
            epe_tensor[eval_id] = image_epe  
            rmse_tensor[eval_id] = (epe_flattened[val]**2).mean().sqrt().item()
            bad1_tensor[eval_id] = b1[val].float().mean().item()
            bad2_tensor[eval_id] = b2[val].float().mean().item()
            bad3_tensor[eval_id] = b3[val].float().mean().item()
            elapsed_tensor[eval_id] = end-start

    epe = epe_tensor.mean().item()
    bad1 = 100 * bad1_tensor.mean().item()
    bad2 = 100 * bad2_tensor.mean().item()
    bad3 = 100 * bad3_tensor.mean().item()
    rmse = rmse_tensor.mean().item()
    avg_runtime = elapsed_tensor.mean().item()
    fps = 1/avg_runtime
    metrics = f"Evaluation {args.dataset}: EPE {epe}, B1 {bad1}, B2 {bad2}, B3 {bad3}, rmse {rmse}, {format(fps, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)"
    print('valid_gru_iters',args.valid_gru_iters)
    print(metrics)

    return {'eval-epe': epe, 'eval-b1': bad1, 'eval-b2': bad2, 'eval-b3': bad3, 'eval-rmse':rmse, 'fps':fps}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dps-net', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="start_epoch")
    parser.add_argument('--end_epoch', type=int, default=100, help="end_epoch")
    parser.add_argument('--start_steps', type=int, default=0, help="start_steps")

    parser.add_argument('--dataset', type=str, default='IPS', help="training datasets.")
    parser.add_argument('--datadir', type=str, default='/mnt/nas_8/datasets/tiancr/b', help="training datalists.")
    parser.add_argument('--training_mode', type=str, default="train", choices=["train", "val", "test", "eval", "debug"])

    parser.add_argument('--reflect_type', type=str, default='rand', help="debug datasets.")

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")


    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")

    parser.add_argument('--train_gru_iters', type=int, default=4, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--valid_gru_iters', type=int, default=4, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=128, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    parser.add_argument('--save_image', action='store_true') 
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)    
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    print("\n### Training shape from polarization model ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    print('restore_ckpt ',args.restore_ckpt)

    eval_dataset = None 
    if args.dataset == 'IPS':
        eval_dataset = datasets.IPS({}, root=args.datadir, reflect_type = "synthetic", mode='test')
    elif args.dataset == 'RPS':
        eval_dataset = datasets.RPS({}, root= args.datadir, reflect_type = "real", mode='test')    
    
    model = nn.DataParallel(DPSNet(args)).cuda()
    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'], strict=False)          
    else:
        model.load_state_dict(checkpoint, strict=False)

    result = eval(args,model, eval_dataset)
