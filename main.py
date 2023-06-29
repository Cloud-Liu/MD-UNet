
import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['MDUNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod

import pdb
from mdrm import *
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.nam=Att(dim,mlp_hidden_dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x1=self.drop_path(self.mlp(self.norm2(x), H, W))
        B, N, C = x.shape
        x2 = x1.transpose(1, 2).view(B, C, H, W)
        x1=self.nam(x2)
        x1 = self.dwconv(x1)
        x1 = x1.flatten(2).transpose(1, 2)

        x = x + x1

        # x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
      
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = torch.sigmoid(x) * residual #
        
        return x


class Att(nn.Module):
    def __init__(self, channels, out_channels=32):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1  

class MDUNet(nn.Module):

    
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        
        self.encoder1=MDRM(3,64,64)
        self.encoder2=MDRM(64,128,128)
        self.encoder3=MDRM(128,256,256)
        self.encoder4=MDRM(256,512,512)
        self.encoder5=MDRM(512,1024,1024)

        self.ebn1 = nn.BatchNorm2d(64)
        self.ebn2 = nn.BatchNorm2d(128)
        self.ebn3 = nn.BatchNorm2d(256)
        self.ebn4 = nn.BatchNorm2d(512)
        self.ebn5 = nn.BatchNorm2d(1024)
        
        
        self.decoder1=MDRM(1024,512,512)
        self.decoder2=MDRM(512,256,256)
        self.decoder3=MDRM(256,128,128)
        self.decoder4=MDRM(128,64,64)
        self.decoder5=MDRM(64,32,32)

        self.dbn1 = nn.BatchNorm2d(512)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(128)
        self.dbn4 = nn.BatchNorm2d(64)
        
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(512, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final2 = nn.Conv2d(256, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final3 = nn.Conv2d(128, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final0 = nn.Conv2d(1024, 32, kernel_size=3,stride=1,padding=1,bias=True)
        

        self.soft = nn.Softmax(dim =1)

        
        self.relu=nn.ReLU(inplace=True)
        

    def forward(self, x):
        
        B = x.shape[0]
        
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
      
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
       
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

       
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        
        t4 = out
        

        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))#torch.Size([2, 1024, 16, 16])
        out0= F.interpolate(self.final(self.final0(out)),scale_factor=(32,32),mode ='bilinear')

       
        

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))#torch.Size([2, 512, 32, 32])
       
        out = torch.add(out,t4)
        # pdb.set_trace()
        out1= F.interpolate(self.final(self.final1(out)),scale_factor=(16,16),mode ='bilinear')
        

       
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out2= F.interpolate(self.final(self.final2(out)),scale_factor=(8,8),mode ='bilinear')
        

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t2)
        out3= F.interpolate(self.final(self.final3(out)),scale_factor=(4,4),mode ='bilinear')
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
       
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return out0,out1,out2,out3,self.final(out)

#MDUNet_M
class MDUNet_M(nn.Module):
    
   
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        
        self.encoder1=MDRM(3,32,32)
        self.encoder2=MDRM(32,64,64)
        self.encoder3=MDRM(64,128,128)
        self.encoder4=MDRM(128,256,256)
        self.encoder5=MDRM(256,512,512)

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)
        self.ebn4 = nn.BatchNorm2d(256)
        self.ebn5 = nn.BatchNorm2d(512)
        
        
        self.decoder1=MDRM(512,256,256)
        self.decoder2=MDRM(256,128,128)
        self.decoder3=MDRM(128,64,64)
        self.decoder4=MDRM(64,32,32)
        self.decoder5=MDRM(32,32,32)

        self.dbn1 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)
        
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(256, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final2 = nn.Conv2d(128, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final3 = nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1,bias=True)
        self.final0 = nn.Conv2d(512, 32, kernel_size=3,stride=1,padding=1,bias=True)
        

        self.soft = nn.Softmax(dim =1)

        
        self.relu=nn.ReLU(inplace=True)
       

    def forward(self, x):
        
        B = x.shape[0]
        
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        
        t4 = out#torch.Size([2, 512, 32, 32])

       

        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))#torch.Size([2, 1024, 16, 16])
        out0= F.interpolate(self.final(self.final0(out)),scale_factor=(32,32),mode ='bilinear')


        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))#torch.Size([2, 512, 32, 32])
       
        out = torch.add(out,t4)
        # pdb.set_trace()
        out1= F.interpolate(self.final(self.final1(out)),scale_factor=(16,16),mode ='bilinear')
        

       
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out2= F.interpolate(self.final(self.final2(out)),scale_factor=(8,8),mode ='bilinear')
       

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t2)
        out3= F.interpolate(self.final(self.final3(out)),scale_factor=(4,4),mode ='bilinear')
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return out0,out1,out2,out3,self.final(out)
#MDUNet_S
class MDUNet_S(nn.Module):
    
   
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        
        self.encoder1=MDRM(3,12,12)
        self.encoder2=MDRM(12,24,24)
        self.encoder3=MDRM(24,32,32)
        self.encoder4=MDRM(32,64,64)
        self.encoder5=MDRM(64,128,128)

        self.ebn1 = nn.BatchNorm2d(12)
        self.ebn2 = nn.BatchNorm2d(24)
        self.ebn3 = nn.BatchNorm2d(32)
        self.ebn4 = nn.BatchNorm2d(64)
        self.ebn5 = nn.BatchNorm2d(128)
        
        
        self.decoder1=MDRM(128,64,64)
        self.decoder2=MDRM(64,32,32)
        self.decoder3=MDRM(32,24,24)
        self.decoder4=MDRM(24,12,12)
        self.decoder5=MDRM(12,12,12)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(24)
        self.dbn4 = nn.BatchNorm2d(12)
        
        
        self.final = nn.Conv2d(12, num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(64, 12, kernel_size=3,stride=1,padding=1,bias=True)
        self.final2 = nn.Conv2d(32, 12, kernel_size=3,stride=1,padding=1,bias=True)
        self.final3 = nn.Conv2d(24, 12, kernel_size=3,stride=1,padding=1,bias=True)
        self.final0 = nn.Conv2d(128, 12, kernel_size=3,stride=1,padding=1,bias=True)
        

        self.soft = nn.Softmax(dim =1)

      
        self.relu=nn.ReLU(inplace=True)
       
    def forward(self, x):
        
        B = x.shape[0]
        
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
      
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
     
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

      
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        
        t4 = out#torch.Size([2, 512, 32, 32])

       

        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))#torch.Size([2, 1024, 16, 16])
        out0= F.interpolate(self.final(self.final0(out)),scale_factor=(32,32),mode ='bilinear')
  

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))#torch.Size([2, 512, 32, 32])
      
        out = torch.add(out,t4)
        # pdb.set_trace()
        out1= F.interpolate(self.final(self.final1(out)),scale_factor=(16,16),mode ='bilinear')
        

       
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out2= F.interpolate(self.final(self.final2(out)),scale_factor=(8,8),mode ='bilinear')
        
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
       
        out = torch.add(out,t2)
        out3= F.interpolate(self.final(self.final3(out)),scale_factor=(4,4),mode ='bilinear')
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
     
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return out0,out1,out2,out3,self.final(out)