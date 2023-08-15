# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2
from datetime import datetime

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, SimpleAugmentor, SparseSimpleAugmentor, SimpleCPSAugmentor, SparseSimpleCPSAugmentor

class PolarStereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None

        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseSimpleAugmentor(**aug_params)

            else:
                self.augmentor = SimpleAugmentor(**aug_params)


        self.is_val = False
        self.log_file = None
        self.init_seed = False
        self.reflect_type = 'synthetic'
        self.disparity_list = []
        self.index_list = []
        self.image_list = []
        self.polar_list = []
        self.valid_list = []
        self.extra_info = []

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.polar_list = v * copy_of_self.polar_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.is_val:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            
            disp = frame_utils.read_gen(self.disparity_list[index])
            disp = np.array(disp).astype(np.float32)
            disp = np.stack([disp], axis=-1)
            disp = torch.from_numpy(disp).permute(2, 0, 1).float()
            
            pol1 = cv2.imread(self.polar_list[index][0],-1)
            pol2 = cv2.imread(self.polar_list[index][1],-1)
            pol1 = np.array(pol1).astype(np.float32)
            pol2 = np.array(pol2).astype(np.float32)

            pol1 = pol1[:,:,:2] / 10000.0
            pol2 = pol2[:,:,:2] / 10000.0  
            pol1 = torch.from_numpy(pol1).permute(2, 0, 1).float()
            pol2 = torch.from_numpy(pol2).permute(2, 0, 1).float()

            valid = frame_utils.read_gen(self.valid_list[index])
            valid = np.array(valid).astype(np.uint8)
            valid = valid > 0
            valid = torch.from_numpy(valid)
            return self.image_list[index], img1, img2, pol1, pol2, disp, valid.float()

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = frame_utils.read_gen(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = (disp < 512) & (disp > 0)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp], axis=-1)

        pol1 =cv2.imread(self.polar_list[index][0],-1)
        pol2 =cv2.imread(self.polar_list[index][1],-1)
        pol1 = np.array(pol1).astype(np.float32)
        pol2 = np.array(pol2).astype(np.float32)
        pol1 = pol1[:,:,:2] / 10000.0
        pol2 = pol2[:,:,:2] / 10000.0  

        valid = frame_utils.read_gen(self.valid_list[index])
        valid = np.array(valid).astype(np.uint8)
        valid = valid > 0

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        clip_offset = []

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, pol1, pol2, flow, valid,clip_offset = self.augmentor(img1, img2, pol1, pol2, flow,valid)
            else:
                img1, img2, pol1, pol2, flow,clip_offset = self.augmentor(img1, img2, pol1, pol2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        pol1 = torch.from_numpy(pol1).permute(2, 0, 1).float()
        pol2 = torch.from_numpy(pol2).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512) & (flow[0]>0)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)
            pol1 = F.pad(pol1, [padW]*2 + [padH]*2)
            pol2 = F.pad(pol2, [padW]*2 + [padH]*2)

        data_list = self.image_list[index] +self.polar_list[index] + [self.disparity_list[index]]
        return data_list, img1, img2, pol1, pol2, flow, valid.float()

class IPS(PolarStereoDataset):
    def __init__(self, aug_params=None, root='/mnt/nas_8/datasets/tiancr/b', reflect_type = 'diff', mode='train'):
        super(IPS, self).__init__(aug_params, sparse=True)
        print(root)
        assert os.path.exists(root)
        print(root)
        self.reflect_type = reflect_type
        if mode=='train':
            self.is_val = False
            list_filename = "filenames/IPS_Dataset_TRAIN.list"
        elif mode=='test':
            self.is_val = True
            list_filename = "filenames/IPS_Dataset_TEST.list"
        elif mode=='eval':
            self.is_val = True
            list_filename = "filenames/IPS_Dataset_TEST.list"

        print("list_filename",list_filename)
        with open(list_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        splits = [line.split() for line in lines]
        image1_list = [os.path.join(root,x[0], 'l_'+x[1]+'.png') for x in splits] # x[0] subset, x[1] index
        image2_list = [os.path.join(root,x[0], 'r_'+x[1]+'.png') for x in splits]

        pol1_list = None
        pol2_list = None        

        if self.reflect_type == 'synthetic':
            pol1_list = [os.path.join(root, 'Render1', x[0], 'clpL_'+x[1]+'.png') for x in splits]
            pol2_list = [os.path.join(root, 'Render1', x[0], 'clpR_'+x[1]+'.png') for x in splits]
        
        self.valid_list = [os.path.join(root, 'Render', x[0], 'v_'+x[1]+'.png') for x in splits]
        self.disparity_list = [os.path.join(root, 'Render', x[0], 'd_'+x[1]+'.pfm') for x in splits]

        for idx, (img1, img2) in enumerate(zip(image1_list, image2_list)):
            self.image_list += [ [img1, img2] ]

        for idx, (lp1, lp2) in enumerate(zip(pol1_list, pol2_list)):
            self.polar_list += [ [lp1, lp2] ]

class RPS(PolarStereoDataset):
    def __init__(self, aug_params=None, root='/mnt/nas_8/datasets/tiancr/b', 
                 with_data_list = True, reflect_type = 'real',
                 mode='train', subset=""):
        super(RPS, self).__init__(aug_params, sparse=True)
        print(root)
        assert os.path.exists(root)
        self.reflect_type = reflect_type        
        if not subset and with_data_list:
            if mode=='train':
                self.is_val = False
                list_filename = "filenames/RPS_Dataset_TRAIN.list"
            elif mode=='test':
                self.is_val = True
                list_filename = "filenames/RPS_Dataset_TEST.list"            
            elif mode=='eval':
                self.is_val = True
                list_filename = "filenames/RPS_Dataset_VAL.list"

            print("list_filename",list_filename)
            if not os.path.exists(list_filename):
                raise FileNotFoundError(list_filename)

            with open(list_filename) as f:
                lines = [line.rstrip() for line in f.readlines()]
            splits = [line.split() for line in lines]        
            
            image1_list = [os.path.join(root,x[0], 'l_'+x[1]+'.png') for x in splits]
            image2_list = [os.path.join(root,x[0], 'r_'+x[1]+'.png') for x in splits]

            pol1_list = None
            pol2_list = None

            if self.reflect_type == 'real': 
                pol1_list = [os.path.join(root, x[0], 'lpL_'+x[1]+'.png') for x in splits]
                pol2_list = [os.path.join(root, x[0], 'lpR_'+x[1]+'.png') for x in splits]

            self.valid_list = [os.path.join(root, x[0], 'v_'+x[1]+'.png') for x in splits]
            self.disparity_list = [os.path.join(root, x[0], 'disp_'+x[1]+'.pfm') for x in splits]

            for idx, (img1, img2) in enumerate(zip(image1_list, image2_list)):
                self.image_list += [ [img1, img2] ]

            for idx, (lp1, lp2) in enumerate(zip(pol1_list, pol2_list)):
                self.polar_list += [ [lp1, lp2] ]
        elif subset and not with_data_list:
            self.is_val=True
            root = os.path.join(root, "PRS")
                                           
            for file in sorted(os.listdir(os.path.join(root, subset))):
                if file.startswith('disp_'):
                    self.disparity_list.append(os.path.join(root, subset, file))
                    self.index_list.append(file[2:-4])
            self.valid_list = [os.path.join(root, subset, 'v_'+x+'.png') for x in self.index_list]
            image1_list = [os.path.join(root,subset, 'l_'+x+'.png') for x in self.index_list]
            image2_list = [os.path.join(root,subset, 'r_'+x+'.png') for x in self.index_list]

            pol1_list = None
            pol2_list = None
            if self.reflect_type == 'real': 
                pol1_list = [os.path.join(root, subset, 'lpL_'+x+'.png') for x in self.index_list]
                pol2_list = [os.path.join(root, subset, 'lpR_'+x+'.png') for x in self.index_list]

            for idx, (img1, img2) in enumerate(zip(image1_list, image2_list)):
                self.image_list += [ [img1, img2] ]

            for idx, (lp1, lp2) in enumerate(zip(pol1_list, pol2_list)):
                self.polar_list += [ [lp1, lp2] ]

            print(subset+'({:d}):'.format(len(self.index_list)), self.index_list)
        elif subset and with_data_list:
            if mode=='train':
                self.is_val = False
                list_filename = "filenames/RPS_Dataset_TRAIN.list"
            elif mode=='test':
                self.is_val = True
                list_filename = "filenames/RPS_Dataset_TEST.list"
            elif mode=='eval':
                self.is_val = True
                list_filename = "filenames/RPS_Dataset_VAL.list"

            print("list_filename",list_filename)
            if not os.path.exists(list_filename):
                raise FileNotFoundError(list_filename)

            with open(list_filename) as f:
                lines = [line.rstrip() for line in f.readlines()]
            splits = [line.split() for line in lines]  

            image1_list = []
            image2_list = []
            pol1_list = []
            pol2_list = []

            for x in splits:                
                if x[0].split('/')[1]==subset:
                    self.index_list.append(x[1])
                    image1_list.append(os.path.join(root,x[0], 'l_'+x[1]+'.png'))
                    image2_list.append(os.path.join(root,x[0], 'r_'+x[1]+'.png'))           
            
                    pol1_list.append(os.path.join(root, x[0], 'lpL_'+x[1]+'.png'))
                    pol2_list.append(os.path.join(root, x[0], 'lpR_'+x[1]+'.png'))

                    self.valid_list.append(os.path.join(root, x[0], 'v_'+x[1]+'.png'))
                    self.disparity_list.append(os.path.join(root, x[0], 'disp_'+x[1]+'.pfm'))

            for idx, (img1, img2) in enumerate(zip(image1_list, image2_list)):
                self.image_list += [ [img1, img2] ]

            for idx, (lp1, lp2) in enumerate(zip(pol1_list, pol2_list)):
                self.polar_list += [ [lp1, lp2] ]

            print(subset+'({:d}):'.format(len(self.index_list)), self.index_list)


def fetch_dataloader(args):
    """ Create the data loader for the corresponding training set """
    aug_params = {'crop_size': args.image_size}

    dataset = None
    if args.dataset == 'IPS':
        new_dataset = IPS(aug_params, root=args.datadir, reflect_type = args.reflect_type)
        logging.info(f"Adding {len(new_dataset)} samples from IPS")
    elif args.dataset == 'RPS':
        new_dataset = RPS(aug_params, root=args.datadir, reflect_type = args.reflect_type)
        logging.info(f"Adding {len(new_dataset)} samples from RPS")

    dataset = new_dataset if dataset is None else dataset + new_dataset
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(dataset))
    logging.info('Training with %d image pairs' % len(train_loader))

    return train_loader
