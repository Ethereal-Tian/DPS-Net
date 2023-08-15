from os import RTLD_GLOBAL
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, head_hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, head_hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(head_hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class DisparityHead(nn.Module):
    def __init__(self, input_dim=256, head_hidden_dim=128):
        super(DisparityHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, head_hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(head_hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class AlphaHead(nn.Module):
    def __init__(self, input_dim=256, head_hidden_dim=128):
        super(AlphaHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, head_hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(head_hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_dim, out_dim):
        super(BasicMotionEncoder, self).__init__()
        self.out_dim = out_dim
        self.convc1 = nn.Conv2d(corr_dim, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, out_dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicCostEncoder(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_dim):
        super(BasicCostEncoder,self).__init__()
        self.out_dim = out_dim
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64+hidden_dim, out_dim - 1, 3, padding=1)
        
    def forward(self, disparity, cost):
        cos = F.relu(self.convc1(cost))
        cos = F.relu(self.convc2(cos))

        dfm = F.relu(self.convd1(disparity))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cos, dfm], dim=1)
        
        out_d = F.relu(self.convd(cor_dfm))
                
        return torch.cat([out_d, disparity], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicOptimizeUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=64, cost_dim=1, ratio=8):
        super(BasicOptimizeUpdateBlock, self).__init__()
                
        self.encoder = BasicCostEncoder(cost_dim=cost_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.disp_gru = ConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_dim+context_dim)
        self.disparity_head = DisparityHead(hidden_dim, head_hidden_dim=hidden_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))


    def forward(self, net, inp, cost, disparity):
        input_features = self.encoder(disparity, cost)
        inp = torch.cat([inp, input_features], dim=1)
        net = self.disp_gru(net, inp)
        delta_disparity = self.disparity_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_disparity

class BasicRGBPCorrUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=128, corr_dim=324):
        super().__init__()
        self.rgb_encoder = BasicMotionEncoder(corr_dim=corr_dim,out_dim=hidden_dim)
        self.pol_encoder = BasicMotionEncoder(corr_dim=corr_dim,out_dim=hidden_dim)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim+self.rgb_encoder.out_dim+self.pol_encoder.out_dim)
        self.flow_head = FlowHead(hidden_dim, head_hidden_dim=256, output_dim=2)
        factor = 8

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, rgb_corr=None, pol_corr=None, flow=None, update=True):
        rgb_motion_features = self.rgb_encoder(flow, rgb_corr)
        pol_motion_features = self.pol_encoder(flow, pol_corr)

        inp = torch.cat([inp, rgb_motion_features,pol_motion_features], dim=1)

        net = self.gru(net, inp)

        if not update:
            return net

        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class BasicOptRectifyUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, context_dim=64, cost_dim=1, corr_dim = 324, ratio=8):
        super().__init__()
        self.cost_encoder = BasicCostEncoder(cost_dim=cost_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.rgb_encoder = BasicMotionEncoder(corr_dim=corr_dim,out_dim=hidden_dim)
        self.pol_encoder = BasicMotionEncoder(corr_dim=corr_dim,out_dim=hidden_dim)
        self.gru = ConvGRU( hidden_dim=hidden_dim, input_dim=context_dim \
                                                + self.rgb_encoder.out_dim+self.pol_encoder.out_dim \
                                                +self.cost_encoder.out_dim)
        self.flow_head = FlowHead(hidden_dim, head_hidden_dim=256, output_dim=2)
        self.alpha_head = AlphaHead(hidden_dim, head_hidden_dim=hidden_dim)
        factor = 8

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, rgb_corr_opt,pol_corr_opt, cost, flow,flow_opt, update=True):
        disparity = flow[:,:1]
        rgb_motion_features_opt = self.rgb_encoder(flow_opt, rgb_corr_opt)
        pol_motion_features_opt = self.pol_encoder(flow_opt, pol_corr_opt)
        input_cost = self.cost_encoder(disparity, cost)

        inp = torch.cat([inp, rgb_motion_features_opt,pol_motion_features_opt,input_cost], dim=1)

        net = self.gru(net, inp)

        if not update:
            return net

        delta_flow = self.flow_head(net)
        alpha = self.alpha_head(net)
        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, alpha, delta_flow

