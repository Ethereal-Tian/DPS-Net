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

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from core.dps_net import DPSNet, autocast

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

def sequence_loss(disp_preds, disparity_gt, valid, loss_gamma=0.9, max_disp=700):
    """ Loss function defined over sequence of disparity predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(disparity_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disparity_gt.shape, [valid.shape, disparity_gt.shape]
    assert not torch.isinf(disparity_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disparity_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disparity_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disparity_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, steps, args):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = steps
        self.running_loss = {}                
        
        if args.training_mode == "train":
            if args.dataset == 'IPS':
                self.log_dir = "./log/ips_" + datetime.now().strftime("%m-%d-%H-%M") + "_" + args.dataset
                setting_file = 'training_setting.txt'
            elif args.dataset == 'RPS':
                self.log_dir = "./log/rps_" + datetime.now().strftime("%m-%d-%H-%M") + "_" + args.dataset
                setting_file = 'finetune_setting.txt'
        else:                        
            self.log_dir = './eval/'
            setting_file = args.dataset + "_gru"+str(args.valid_gru_iters)+"_all.txt"    
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print('log file saved in', self.log_dir)   
        args.log_file = os.path.join(self.log_dir, setting_file)     
            
        with open(args.log_file, 'w') as f:
            if args.dataset == 'RPS' and args.training_mode == 'train':
                f.write('finetune IPS model: '+args.restore_ckpt+'\n')
            if args.training_mode != "train":
                f.write('restore ckpt from: '+args.restore_ckpt+'\n')                                    
            f.write("train_gru_iters: "+str(args.train_gru_iters)+'\n')
            f.write("valid_gru_iters: "+str(args.valid_gru_iters)+'\n')
            f.write("start_epoch: "+str(args.start_epoch)+'\n')
            f.write("end_epoch: "+str(args.end_epoch)+'\n')
            f.write("start_steps: "+str(args.start_steps)+'\n')
            f.write("num_steps: "+str(args.num_steps)+'\n')
            f.write("max learning rate: "+str(args.lr)+'\n')
            f.write("batch_size: "+str(args.batch_size)+'\n')
            f.write("gpu id: "+os.environ["CUDA_VISIBLE_DEVICES"]+'\n')
            f.write("save root: "+'%s' % (args.save_root)+'\n')
            
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        pnum_str = "[{:6d}] ".format(self.predictions_num)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str+pnum_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}
    
    def set_predictions_num(self,predictions_n):
        self.predictions_num = predictions_n

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args):
    model = nn.DataParallel(DPSNet(args)).cuda()
    
    if args.dataset == 'RPS':
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint['model'], strict=True)  

    train_loader = datasets.fetch_dataloader(args)   
    val_dataset = None
    if args.dataset == 'IPS':
        val_dataset = datasets.IPS({}, root=args.datadir, reflect_type = "synthetic", mode='eval')
    elif args.dataset == 'RPS':
        val_dataset = datasets.RPS({}, root= args.datadir, reflect_type = "real", mode='eval')    

    total_steps = args.start_steps    
    save_frequency = 1   
    validate_frequency = 1
    epe_min = 100.0
    best_val_path = Path('%s/best_val_%s.pth' % (args.save_root,  args.dataset))


    optimizer, scheduler = fetch_optimizer(args, model)
    epoch0 = args.start_epoch
    epochn = args.end_epoch
    steps = args.start_steps
        
    logger = Logger(model, scheduler, total_steps, args=args)
    
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)
    
    for epoch in range(epoch0, epochn):                    
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            imgL, imgR, polL, polR, disp_gt, valid = [x.cuda() for x in data_blob]
            disp_predictions = model(imgL, imgR, polL, polR, gru_iters=args.train_gru_iters)
            loss, metrics = sequence_loss(disp_predictions, disp_gt, valid)
            logger.writer.add_scalar("live_loss", loss.item(), steps)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], steps)
            steps += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            logger.set_predictions_num(len(disp_predictions))
            
        if epoch % save_frequency == save_frequency - 1:
            save_path = Path('%s/%s_epoch_%d.pth' % (args.save_root, args.dataset, epoch))
            logging.info(f"Saving file {save_path.absolute()} ({epoch}/{args.end_epoch})")
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'steps': steps-1
            }
            torch.save(checkpoint, save_path)
        
        if args.dataset=='RPS' and epoch % validate_frequency == 0:
            result = val(model, val_dataset, args)
            logger.write_dict(result)
            if(result['val-epe']<epe_min):
                epe_min = result['val-epe']
                logging.info(f"Saving best val file {best_val_path.absolute()}")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'steps': steps-1
                }
                torch.save(checkpoint, best_val_path)
                print('best ckpt',best_val_path)
            
        model.train()
        model.module.freeze_bn()

    print("FINISHED TRAINING, min epe =", epe_min)
    logger.close()

    PATH = '%s/%d_%s.pth' % (args.save_root, args.end_epoch, datetime.now().strftime("%m-%d-%H-%M"))
    torch.save(model.state_dict(), PATH)
    return PATH

def val(model, val_dataset, args):
    model.eval()
    torch.backends.cudnn.benchmark = True
    out_tensor = torch.zeros(len(val_dataset)).cuda().float()
    epe_tensor = torch.zeros(len(val_dataset)).cuda().float()
    elapsed_tensor = torch.zeros(len(val_dataset)).cuda().float()

    for val_id in tqdm(range(len(val_dataset))):
        _, imgL, imgR, polL, polR, disp_gt, valid_gt = val_dataset[val_id]
        
        imgL = imgL[None].cuda()
        imgR = imgR[None].cuda()
        polL = polL[None].cuda()
        polR = polR[None].cuda()
        disp_gt = disp_gt[None].cuda().squeeze(0)

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

        out = (epe_flattened > 3.0) #3px
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        epe_tensor[val_id] = image_epe
        out_tensor[val_id] = image_out
        elapsed_tensor[val_id] = end-start
    
    epe = epe_tensor.mean().item()
    d1 = 100 * out_tensor.mean().item()
    avg_runtime = elapsed_tensor.mean().item()

    print(f"Validation {args.dataset}: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'val-epe': epe, 'val-d1': d1}


def eval(args):
    assert os.path.exists(args.output_path)
    model = nn.DataParallel(DPSNet(args)).cuda()
    checkpoint = torch.load(args.restore_ckpt)
    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'], strict=False)          
    else:
        model.load_state_dict(checkpoint, strict=False)

    logger = Logger(model, None, None, args=args)


    eval_dataset = None 
    if args.dataset == 'IPS':
        eval_dataset = datasets.IPS({}, root=args.datadir, reflect_type = "synthetic", mode='eval')
    elif args.dataset == 'RPS':
        eval_dataset = datasets.RPS({}, root= args.datadir, reflect_type = "real", mode='eval')    
    model.eval()
    torch.backends.cudnn.benchmark=True


    bad1_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    bad2_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    bad3_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    epe_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    elapsed_tensor = torch.zeros(len(eval_dataset)).cuda().float()
    invalid_img_num = 0
    invalid_img_id = []
    
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
            bad1_tensor[eval_id] = b1[val].float().mean().item()
            bad2_tensor[eval_id] = b2[val].float().mean().item()
            bad3_tensor[eval_id] = b3[val].float().mean().item()
            elapsed_tensor[eval_id] = end-start
                
    epe = epe_tensor.mean().item()
    bad1 = 100 * bad1_tensor.mean().item()
    bad2 = 100 * bad2_tensor.mean().item()
    bad3 = 100 * bad3_tensor.mean().item()
    avg_runtime = elapsed_tensor.mean().item()
    fps = 1/avg_runtime
    metrics = f"Evaluation {args.dataset}: EPE {epe}, B1 {bad1}, B2 {bad2}, B3 {bad3}, {format(fps, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)"
    print(metrics)
    with open(args.log_file, "a") as f:
        f.write(metrics+"\n")
        f.write("invalid image: "+str(invalid_img_num)+"/"+str(len(eval_dataset))+"\n")
        f.write(str(invalid_img_id)+"\n")

    return {'eval-epe': epe, 'eval-b1': bad1, 'eval-b2': bad2, 'eval-b3': bad3, 'fps':fps}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='dps-net', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="start_epoch")
    parser.add_argument('--end_epoch', type=int, default=100, help="end_epoch")
    parser.add_argument('--start_steps', type=int, default=0, help="start_steps")

    parser.add_argument('--dataset', type=str, default='IPS', help="training datasets.")
    parser.add_argument('--datadir', type=str, default='/mnt/nas_8/datasets/tiancr/b', help="training datalists.")
    parser.add_argument('--training_mode', type=str, default="train", choices=["train", "test", "eval"])
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

    parser.add_argument('--save_root', default="./checkpoints/checkpoint_"+datetime.now().strftime("%m%d"), type=str)   
    parser.add_argument('--output_path', type=str, default="") 

    args = parser.parse_args()
    args.save_root = args.save_root + "_" + args.dataset
    torch.manual_seed(1234)
    np.random.seed(1234)    
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # Path("checkpointy").mkdir(exist_ok=True, parents=True)
    if args.training_mode == 'train':
        Path('%s' % (args.save_root)).mkdir(exist_ok=True, parents=True)

    print("\n### Training shape from polarization model ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    if args.training_mode == 'train':
        train(args)
    elif args.training_mode == 'eval':
        args.output_path = args.output_path + datetime.now().strftime("_%m%d")
        os.makedirs(args.output_path, exist_ok=True)
        eval(args)