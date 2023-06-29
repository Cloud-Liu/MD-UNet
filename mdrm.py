
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d
from torch.nn import init
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from timm.models.vision_transformer import _cfg
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    

class MixShiftBlock(nn.Module):
    r""" Mix-Shifting Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size, layer_scale_init_value=1e-6,
                 mlp_ratio=4, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.shift_size = shift_size
        self.shift_dist = shift_dist
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]
        
        self.kernel_size = [(ms, ms//2) for ms in mix_size]
        self.dwconv_lr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size = kernel_size[0], padding = kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        self.dwconv_td = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, kernel_size = kernel_size[0], padding = kernel_size[1], groups=chunk_dim) for chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim)) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.nam=Att(dim,dim)

    def forward(self, x):
        input = x
        B_, C, H, W = x.shape

        # split groups
        xs = torch.chunk(x, self.shift_size, 1) 

        # shift with pre-defined relative distance
        x_shift_lr = [ torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
        x_shift_td = [ torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]

       
        for i in range(self.shift_size):
            # pdb.set_trace()
           
            x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
            # print('x.shape:')
            # print(x.shape)
            # print('len(xs):')
            # print(len(xs))
            
            x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])

        x_lr = torch.cat(x_shift_lr, 1)
        x_td = torch.cat(x_shift_td, 1)
        
        x = x_lr + x_td 
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        # x=self.nam(x)

        x = input + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        # dwconv_1 dwconv_2
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        # x_lr + x_td
        flops += N * self.dim
        # norm
        flops += self.dim * H * W
        # pwconv
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops


class MDAB(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=3, dw_size=3, stride=1, relu=True): #ratio=2 default, ratio=6 for gpunet
        super(MDAB, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)
      
        self.conv = nn.Sequential(
           
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
          
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.conv_1x1_output = nn.Conv2d((init_channels * 4), oup, 1, 1)

        
      

        self.msmlp=MixShiftBlock(dim=init_channels, input_resolution=224,
                              shift_size=4,#4
                              shift_dist=[-2,-1,0,1,2],
                              mix_size=[1,1,3,5,7],
                              mlp_ratio=4.,
                              drop=0., 
                              drop_path=0.,
                              norm_layer=nn.LayerNorm)




   
    def forward(self, x):
        # pdb.set_trace()
        x1 = self.conv(x)
        

        x2 = self.cheap_operation(x1)
       
        xmsmlp=self.msmlp(x1)
      
        out = self.conv_1x1_output(torch.cat([x1,x2,xmsmlp], dim=1))
        
        return out[:,:self.oup,:,:]






class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class MDRM(nn.Module):
   

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.25):
        super(MDRM, self).__init__()
        # pdb.set_trace
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        
       
        self.mdab1 = MDAB(in_chs, mid_chs, relu=True)#MD-UNet
        
        
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        
        self.mdab2 = MDAB(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        # pdb.set_trace()
        residual = x

        # 1st ghost bottleneck
        x = self.mdab1(x)

        
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.mdab2(x)
        
        x += self.shortcut(residual)
        return x



