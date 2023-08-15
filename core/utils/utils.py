import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)


def laplacian_filter(input):

    laplacian_kernel = torch.tensor([[1.0/12.0, 1.0/6.0, 1.0/12.0],
                                                                    [ 1.0/6.0,    -1.0, 1.0/6.0],
                                                                    [1.0/12.0, 1.0/6.0, 1.0/12.0]])
    laplacian_kernel = laplacian_kernel.view(1,1,3,3).to(input)

    B, D, H, W = input.shape
    output = F.conv2d(input.reshape(B*D,1,H,W), laplacian_kernel, padding=1)
    return output.view(B, D, H, W)

def neighbour_filter(input):
    input = input.int()
    neighbour_kernel = torch.tensor([   [ 0, 1, 0],
                                        [ 1, 1, 1],
                                        [ 0, 1, 0]])
    neighbour_kernel = neighbour_kernel.view(1,1,3,3).to(input)

    B, D, H, W = input.shape
    output = F.conv2d(input.reshape(B*D,1,H,W), neighbour_kernel, padding=1).view(B, D, H, W)
    return (output == 5) 

def neighbour_filter2(input):
    input = input.float()
    neighbour_kernel = torch.tensor([       [ 1.0, 1.0, 1.0],
                                            [ 1.0, 1.0, 1.0],
                                            [ 1.0, 1.0, 1.0]])
    neighbour_kernel = neighbour_kernel.view(1,1,3,3).to(input)
    B, D, H, W = input.shape
    output = F.conv2d(input.reshape(B*D,1,H,W), neighbour_kernel, padding=1).view(B, D, H, W)
    return (output > 8.0) 


def dxdy_calc(input):
    #dx d(x+1,y)−d(x,y)
    #dy d(x,y+1)−d(x,y)
    dx = torch.empty(input.shape,device = input.device)
    dy = torch.empty(input.shape,device = input.device)

    dx[:,:,:,:-1] = input[:,:,:,1:] - input[:,:,:,:-1]
    dx[:,:,:,-1] = 0
    
    dy[:,:,:-1,:] = input[:,:,1:,:] - input[:,:,:-1,:]
    dy[:,:,-1,:] = 0    

    return dx,dy

def mask_in_range(input,valid_mask):
    inp_zero = torch.zeros_like(input)
    input_mask_zero = torch.where(valid_mask,input,inp_zero)
    float_mask = valid_mask.float()
    input_avg = torch.sum(input_mask_zero) / torch.sum(float_mask)
    
    inp_avg = torch.ones_like(input) * input_avg
    input_mask_avg = torch.where(valid_mask,input,inp_avg)

    input_max = torch.max(input_mask_avg)
    input_min = torch.min(input_mask_avg)

    return input_max,input_min,input_avg

def modify_by_mask(input, valid_mask, modify_val):
    inp_modify = torch.ones_like(input) * modify_val
    output = torch.where(valid_mask,input,inp_modify)
    # print('valid_mask',valid_mask.shape)
    # print('input',input.shape)
    # print('output',output.shape)
    return output

