from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import DisparityHead,UpMaskNet,BasicRGBPCorrUpdateBlock,BasicOptimizeUpdateBlock,BasicOptRectifyUpdateBlock
from core.extractor import BasicEncoder, BasicEncoderV2, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8, neighbour_filter2, modify_by_mask
import math
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class DPSNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        feat_ratio = 8
        
        self.rgb_fout_dim = 256
        self.pol_fout_dim = 256
        self.opt_fout_dim = 128

        self.rgb_hidden_dim = args.hidden_dim
        self.pol_hidden_dim = args.hidden_dim
        self.opt_hidden_dim = 128 # if self.is_high else 64
        
        self.rgb_context_dim = args.hidden_dim
        self.pol_context_dim = args.hidden_dim
        self.opt_context_dim = 32
        self.cost_dim = 1

        self.corr_dim = args.corr_levels * (2*args.corr_radius + 1) #4*9

        self.rgb_cnet = BasicEncoderV2(input_dim = 3,output_dim=[self.rgb_hidden_dim, self.rgb_context_dim], norm_fn="batch")
        self.pol_cnet = BasicEncoderV2(input_dim = 2,output_dim=[self.pol_hidden_dim, self.pol_context_dim], norm_fn="batch")
        self.opt_cnet = BasicEncoderV2(input_dim = 3,output_dim=[self.rgb_hidden_dim, self.rgb_context_dim], norm_fn="batch")
        
        self.opt_rect_update_block = BasicOptRectifyUpdateBlock(hidden_dim=self.rgb_hidden_dim, context_dim = self.rgb_context_dim, cost_dim = self.cost_dim, corr_dim=self.corr_dim, ratio=feat_ratio)
        self.rgb_pol_update_block = BasicRGBPCorrUpdateBlock(hidden_dim=self.rgb_hidden_dim, context_dim = self.rgb_context_dim, corr_dim=self.corr_dim)

        if args.shared_backbone:
            self.rgb_conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, self.rgb_fout_dim, 3, padding=1))
            self.pol_conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, self.pol_fout_dim, 3, padding=1))
            self.opt_conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, self.pol_fout_dim, 3, padding=1))
        else:
            self.rgb_fnet = BasicEncoder(input_dim = 3,output_dim=self.rgb_fout_dim, norm_fn='instance')
            self.pol_fnet = BasicEncoder(input_dim = 2,output_dim=self.pol_fout_dim, norm_fn='instance')
            self.opt_fnet = BasicEncoder(input_dim = 3,output_dim=self.opt_fout_dim, norm_fn='instance')

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 8
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    def upsample_disparity(self, disparity, mask):
        """ Upsample disparity field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = disparity.shape
        factor = 8
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disparity = F.unfold(disparity, [3,3], padding=1)
        up_disparity = up_disparity.view(N, 1, 9, 1, 1, H, W)

        up_disparity = torch.sum(mask * up_disparity, dim=2)
        up_disparity = up_disparity.permute(0, 1, 4, 2, 5, 3)
        return up_disparity.reshape(N, 1, factor*H, factor*W)  

    def aolp_cost_calc(self, pol_l, pol_r, disparity):
        B, _, H, W = disparity.shape

        fx = 400
        fy = 400

        if self.args.dataset == 'RPS':
            fx = 1192.7164074952859
            fy = 1192.7108141046760
        elif self.args.dataset == 'IPS':
            fx = 480
            fy = 480

        min_disp = 0.01
        max_disp = 200

        dolp_l = pol_l[:,0,:,:].unsqueeze(1)
        aolp_l = pol_l[:,1,:,:].unsqueeze(1)
        aolp_l = (aolp_l + 1.0) * math.pi / 2.0

        valid_mask = (disparity > min_disp) &  (disparity < max_disp)
        
        neigh_mask = neighbour_filter2(valid_mask)
        
        valid_mask = valid_mask & neigh_mask

        ax = torch.zeros_like(disparity, device = disparity.device)
        ay = torch.zeros_like(disparity, device = disparity.device)

        dx0l = torch.zeros_like(disparity, device = disparity.device)
        dx0r = torch.zeros_like(disparity, device = disparity.device)
        dxu0 = torch.zeros_like(disparity, device = disparity.device)
        dxd0 = torch.zeros_like(disparity, device = disparity.device)    
        dy0l = torch.zeros_like(disparity, device = disparity.device)
        dy0r = torch.zeros_like(disparity, device = disparity.device)
        dyu0 = torch.zeros_like(disparity, device = disparity.device)
        dyd0 = torch.zeros_like(disparity, device = disparity.device)

        ax[:,:,1:-1,1:-1] = fx * (disparity[:,:,1:-1,:-2] - disparity[:,:,1:-1,2:]) * (disparity[:,:,:-2,1:-1] + disparity[:,:,2:,1:-1]) / (fx*fy)
        ay[:,:,1:-1,1:-1] = fy *  (disparity[:,:,1:-1,:-2] + disparity[:,:,1:-1,2:]) * (disparity[:,:,:-2,1:-1] - disparity[:,:,2:,1:-1]) / (fx*fy)

        dx0l[:,:,1:-1,1:-1] = fx * (1.0) * (disparity[:,:,:-2,1:-1] + disparity[:,:,2:,1:-1]) / (fx*fy) #[:,:,1:-1,:-2]
        dx0r[:,:,1:-1,1:-1] = fx * (- 1.0) * (disparity[:,:,:-2,1:-1] + disparity[:,:,2:,1:-1]) / (fx*fy) #[:,:,1:-1,2:]
        dxu0[:,:,1:-1,1:-1] = fx * (1.0) * (disparity[:,:,1:-1,:-2] - disparity[:,:,1:-1,2:]) / (fx*fy) #[:,:,:-2,1:-1]
        dxd0[:,:,1:-1,1:-1] = fx * (1.0) * (disparity[:,:,1:-1,:-2] - disparity[:,:,1:-1,2:]) / (fx*fy) #[:,:,2:,1:-1]
        dy0l[:,:,1:-1,1:-1] = fy * (1.0) * (disparity[:,:,:-2,1:-1] - disparity[:,:,2:,1:-1]) / (fx*fy) #[:,:,1:-1,:-2]
        dy0r[:,:,1:-1,1:-1] = fy * (1.0) * (disparity[:,:,:-2,1:-1] - disparity[:,:,2:,1:-1]) / (fx*fy) #[:,:,1:-1,2:]
        dyu0[:,:,1:-1,1:-1] = fy * (1.0) * (disparity[:,:,1:-1,:-2] + disparity[:,:,1:-1,2:]) / (fx*fy) #[:,:,:-2,1:-1]
        dyd0[:,:,1:-1,1:-1] = fy * (-1.0) * (disparity[:,:,1:-1,:-2] + disparity[:,:,1:-1,2:]) / (fx*fy) #[:,:,2:,1:-1]

        diffuse_azimuth_cost = torch.pow( torch.sin(aolp_l) * ax - torch.cos(aolp_l) * ay , 2)
        specular_azimuth_cost = torch.pow( torch.sin(aolp_l) * ay + torch.cos(aolp_l) * ax , 2)

        diff_cost_x = 2.0 * ( torch.sin(aolp_l) * ax - torch.cos(aolp_l) * ay) * torch.sin(aolp_l)
        diff_cost_y = -2.0* ( torch.sin(aolp_l) * ax - torch.cos(aolp_l) * ay ) * torch.cos(aolp_l)
        spec_cost_x =2.0*  ( torch.sin(aolp_l) * ay + torch.cos(aolp_l) * ax) *torch.cos(aolp_l)
        spec_cost_y = 2.0* ( torch.sin(aolp_l) * ay + torch.cos(aolp_l) * ax) * torch.sin(aolp_l)

        reflect_none = -1.0 * torch.ones_like(disparity).float()
        reflect_spec = 0.0 * torch.ones_like(disparity).float()
        reflect_diff = 1.0 * torch.ones_like(disparity).float()

        ref_type = torch.where(diffuse_azimuth_cost < specular_azimuth_cost, reflect_diff, reflect_spec)
        ref_type = torch.where((dolp_l >= 0.5) & (dolp_l <= 1.0), reflect_spec, ref_type)
        ref_type = torch.where(((dolp_l == 0.0) & (aolp_l== 0.0)) | (dolp_l > 1.0) |  (valid_mask == False), reflect_none, ref_type)
        
        cost =  0.0 * torch.ones_like(disparity).float()
        cost = torch.where( ref_type == 0.0, specular_azimuth_cost, cost)
        cost = torch.where( ref_type == 1.0, diffuse_azimuth_cost, cost)

        cost_x = 0.0 * torch.ones_like(disparity).float()
        cost_x = torch.where( ref_type == 0.0,spec_cost_x, cost_x)
        cost_x = torch.where( ref_type == 1.0,diff_cost_x, cost_x)

        cost_y = 0.0 * torch.ones_like(disparity).float()
        cost_y = torch.where( ref_type == 0.0,spec_cost_y, cost_y)
        cost_y = torch.where( ref_type == 1.0,diff_cost_y, cost_y)

        cost = cost * dolp_l
        cost_x = cost_x * dolp_l
        cost_y = cost_y * dolp_l

        disp_grad = torch.zeros_like(disparity, device = disparity.device)
        disp_grad[:,:,1:-1,1:-1] =   cost_x[:,:,1:-1,2:] * dx0l[:,:,1:-1,2:] + \
                                                            cost_y[:,:,1:-1,2:] * dy0l[:,:,1:-1,2:] + \
                                                            cost_x[:,:,1:-1,:-2] * dx0r[:,:,1:-1,:-2] + \
                                                            cost_y[:,:,1:-1,:-2] * dy0r[:,:,1:-1,:-2] + \
                                                            cost_x[:,:,2:,1:-1] * dxu0[:,:,2:,1:-1] + \
                                                            cost_y[:,:,2:,1:-1] * dyu0[:,:,2:,1:-1] + \
                                                            cost_x[:,:,:-2,1:-1] * dxd0[:,:,:-2,1:-1] + \
                                                            cost_y[:,:,:-2,1:-1] * dyd0[:,:,:-2,1:-1]

        disp_grad[:,:,0,1:-1] =    cost_x[:,:,1,1:-1] * dxu0[:,:,1,1:-1] + cost_y[:,:,1,1:-1] * dyu0[:,:,1,1:-1]
        disp_grad[:,:,-1,1:-1] =   cost_x[:,:,-2,1:-1] * dxd0[:,:,-2,1:-1] + cost_y[:,:,-2,1:-1] * dyd0[:,:,-2,1:-1]
        disp_grad[:,:,1:-1,0] =   cost_x[:,:,1:-1,1] * dx0l[:,:,1:-1,1] + cost_y[:,:,1:-1,1] * dy0l[:,:,1:-1,1]
        disp_grad[:,:,1:-1,-1] =   cost_x[:,:,1:-1,-2] * dx0r[:,:,1:-1,-2] + cost_y[:,:,1:-1,-2] * dy0r[:,:,1:-1,-2]

        assert not torch.isnan(cost).any() and not torch.isinf(cost).any()
        cost = F.avg_pool2d(cost,8)

        return cost,disp_grad

    def forward(self, rgb_img_l, rgb_img_r, pol_img_l, pol_img_r, gru_iters=12, disparity_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        rgb_img_l = (2 * (rgb_img_l / 255.0) - 1.0).contiguous()
        rgb_img_r = (2 * (rgb_img_r / 255.0) - 1.0).contiguous()

        # dolp normalize
        pol_img_l[:,1,:,:] = (2 * (pol_img_l[:,1,:,:] / math.pi) - 1.0)
        pol_img_r[:,1,:,:] = (2 * (pol_img_r[:,1,:,:] / math.pi) - 1.0)
        pol_img_l = pol_img_l.contiguous()
        pol_img_r = pol_img_r.contiguous()

        #########  Initalize rgb Feature Map,Context Map,Hidden Map,Input Map ###########
        # run the rgb context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                rgb_cnet_list, rgb_x = self.rgb_cnet(torch.cat((rgb_img_l, rgb_img_r), dim=0), dual_inp=True)
                rgb_fmap1, rgb_fmap2 = self.rgb_conv2(rgb_x).split(dim=0, split_size=rgb_x.shape[0]//2)
            else:
                rgb_cnet_list = self.rgb_cnet(rgb_img_l)
                rgb_fmap1, rgb_fmap2 = self.rgb_fnet([rgb_img_l, rgb_img_r])
            rgb_net = torch.tanh(rgb_cnet_list[0])
            rgb_inp = torch.relu(rgb_cnet_list[1])
    

        #########  Initalize polarimetric Feature Map,Context Map,Hidden Map,Input Map ###########
        # run the rgb context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *pol_cnet_list, pol_x = self.pol_cnet(torch.cat((pol_img_l, pol_img_r), dim=0), dual_inp=True)
                pol_fmap1, pol_fmap2 = self.pol_conv2(pol_x).split(dim=0, split_size=pol_x.shape[0]//2)
            else:
                pol_fmap1, pol_fmap2 = self.pol_fnet([pol_img_l, pol_img_r])
        
        #########  Initalize  Correlation Volume  ###########
        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            rgb_fmap1, rgb_fmap2 = rgb_fmap1.float(), rgb_fmap2.float()
            pol_fmap1, pol_fmap2 = pol_fmap1.float(), pol_fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            rgb_fmap1, rgb_fmap2 = rgb_fmap1.float(), rgb_fmap2.float()
            pol_fmap1, pol_fmap2 = pol_fmap1.float(), pol_fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        rgb_corr_fn = corr_block(rgb_fmap1, rgb_fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        pol_corr_fn = corr_block(pol_fmap1, pol_fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

       #########  Initalize polarimetric Feature Map,Context Map,Hidden Map,Input Map ###########
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                opt_cnet_list, opt_x = self.opt_cnet(torch.cat((rgb_img_l, rgb_img_r), dim=0), dual_inp=True)
                opt_fmap1, opt_fmap2 = self.opt_conv2(opt_x).split(dim=0, split_size=opt_x.shape[0]//2)
            else:
                opt_cnet_list = self.opt_cnet(rgb_img_l)
                opt_fmap1, opt_fmap2 = self.opt_fnet([rgb_img_l, rgb_img_r])
            opt_net = torch.tanh(opt_cnet_list[0])
            opt_inp = torch.relu(opt_cnet_list[1])
        
        #########  Initalize polarimetric Feature Map,Context Map,Hidden Map,Input Map ###########
        # initial disparity
        coords0, coords1 = self.initialize_flow(rgb_net)
        flow_init = torch.zeros(opt_fmap1.shape[0],2,opt_fmap1.shape[2],opt_fmap1.shape[3],device = opt_fmap1.device)

        if disparity_init is None:
            disparity_init = torch.ones(rgb_img_l.shape[0],1,int(rgb_img_l.shape[2] / 8),int(rgb_img_l.shape[3] / 8),device = opt_fmap1.device)

        flow_init[:,:1] = -disparity_init[:,:1]
        coords1 = coords1 + flow_init
        
        disparity_predictions = []
        disparity_up_init = torch.ones(rgb_img_l.shape[0],1,rgb_img_l.shape[2],rgb_img_l.shape[3],device = rgb_img_l.device)
        
        flow = flow_init
        disparity = disparity_init
        disparity_up = disparity_up_init
        alpha = torch.zeros_like(disparity)

        for gru_iter in range(gru_iters):
            flow = flow.detach()
            disparity = disparity.detach()
            disparity_up = disparity_up.detach()
            alpha = alpha.detach()

            coords1 = coords0 + flow
            rgb_corr = rgb_corr_fn(coords1) # index correlation volume
            pol_corr = pol_corr_fn(coords1) # index correlation volume

            with autocast(enabled=self.args.mixed_precision):
                rgb_net, up_mask, delta_flow = self.rgb_pol_update_block(rgb_net, rgb_inp, rgb_corr, pol_corr, flow)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0
            # F(t+1) = F(t) + \Delta(t)
            flow = flow + delta_flow
            disparity = -flow[:,:1]

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(flow)
            else:
                flow_up = self.upsample_flow(flow, up_mask)
            disparity_up = -flow_up[:,:1]
            assert not torch.isnan(disparity_up).any() and not torch.isinf(disparity_up).any()

            if not test_mode:
                disparity_predictions.append(disparity_up)
                
            flow = flow.detach()
            disparity = disparity.detach()
            disparity_up = disparity_up.detach()
            alpha = alpha.detach()

            polar_cost, disp_grad = self.aolp_cost_calc(pol_img_l,pol_img_r,disparity_up)
            cost_grad = F.avg_pool2d(disp_grad,8)

            polar_cost = polar_cost.detach()
            cost_grad = cost_grad.detach()

            disparity_opt = disparity  - cost_grad * alpha
            disparity_opt = torch.where( disparity_opt < 0.0, torch.ones_like(disparity), disparity_opt)
            disparity_opt = disparity_opt.detach()
                
            flow_opt = flow_init
            flow_opt[:,:1] = -disparity_opt
            flow_opt[:,1] = 0.0
            flow_opt = flow_opt.detach()

            coords_opt = coords0 + flow_opt
            rgb_corr_opt = rgb_corr_fn(coords_opt) # index correlation volume
            pol_corr_opt = pol_corr_fn(coords_opt) # index correlation volume
            with autocast(enabled=self.args.mixed_precision):
                opt_net, opt_up_mask, alpha, delta_flow = self.opt_rect_update_block(opt_net, opt_inp, rgb_corr_opt, pol_corr_opt, polar_cost, flow,flow_opt)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            flow = flow + delta_flow
            disparity = -flow[:,:1]

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(flow)
            else:
                flow_up = self.upsample_flow(flow, up_mask)
            disparity_up = -flow_up[:,:1]
            assert not torch.isnan(disparity_up).any() and not torch.isinf(disparity_up).any()

            if not test_mode:
                disparity_predictions.append(disparity_up)

            if test_mode and gru_iter == gru_iters-1:
                disparity_predictions.append(disparity_up)
    
        if test_mode:
            return coords1 - coords0, disparity_up
        return disparity_predictions