import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import numbers

from functools import partial

from utils import wavelet

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
######################################################
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )
        #self.mlp = FeedForward_out(dim, ffn_expansion_factor,bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))#x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#x = x + self.mlp(self.norm2(x))
        return x
#############################################################

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)   #ReLU6?leakyrelu?
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity,
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):  #5#2
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, 1, bias=False)#nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        #self.deconv = DEConv(dim=med_channels,kernel_size=kernel_size,padding=padding)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(med_channels, dim, 1, bias=False)#nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        res = x
        x = self.pwconv1(x)
        x = self.act1(x)
        #x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        #x = self.deconv(x)
        #x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x + res

class Mlp_meta(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=SquaredReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1,bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Conv2d(hidden_features, out_features,kernel_size=1, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        #x = self.drop2(x)
        return x

class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, mlp_ratio=4, act_layer=SquaredReLU,#nn.GELU,SquaredReLU
        head_dropout=0.):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden_features, 1, bias=False)#nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = LayerNorm(hidden_features, 'WithBias')
        self.fc2 = nn.Conv2d(hidden_features, dim, 1, bias=False)#nn.Linear(hidden_features, dim, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.head_dropout(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)

        return x

class ConvFormer(nn.Module):
    def __init__(self, dim, drop_path=0.1,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = LayerNorm(dim, 'WithBias')
        self.token_mixer = SepConv(dim)  # vits是msa，MLPs是mlp，这个用pool来替代
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = MlpHead(dim=dim) #Mlp_meta  FeedForward_Duals
        #self.mlp = FeedForward_Duals(dim=dim, ffn_expansion_factor=2, bias=bias)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale

        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim, dtype=torch.float32) * layer_scale_init_value)

            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim, dtype=torch.float32) * layer_scale_init_value)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#############################################################################


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),  #LeakyReLU
            # dw
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            #detail enhancement
            DEConv(hidden_dim),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
        #self.bottleneckBlock = DEBlockTrain(default_conv,dim=inp,kernel_size=3)

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

##################

class DEABlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(DEABlockTrain, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)#1
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class ChannelAttention_am(nn.Module):
    def __init__(self, in_planes, reduction=8):
        super(ChannelAttention_am, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class DEBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlockTrain, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res

class GLFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim*2, reduction) #ChannelAttention_am
        self.pa = PixelAttention(dim*2)
        self.conv = nn.Conv2d(dim*2, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.mish = nn.Mish()

    def forward(self, x, y):
        initial = torch.cat([x,y],dim=1)#x + y
        cattn = self.ca(initial)#initial
        sattn = self.sa(initial)#initial
        pattn1 = sattn + cattn
        #pattn1 = torch.cat([cattn,sattn],dim=1)
        pattn2 = self.gelu(self.pa(initial, pattn1))#sigmoid gelu mish
        #pattn2 = self.pa(initial, pattn1)
        result = initial * pattn2  #glue
        #result = self.gelu(initial) * pattn2
        return self.conv(result + initial)
        #result = initial + pattn2 * x + (1 - pattn2) * y
        #result = self.conv(result)

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, 1:]
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -conv_weight[:, :, 1:] * self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 0] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias,
                                            stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res

#========================================
class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample_factor):
        super(UpSampleConv, self).__init__()
        self.upsample_factor = upsample_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # 先使用最近邻插值或双线性插值进行上采样雙三
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode='bicubic', align_corners=False)
        # 然后通过卷积进行平滑
        x = self.conv(x)
        return x

#=====================================================
class SubPixelCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SubPixelCNN, self).__init__()

        # 进行卷积，增加通道数为 upscale_factor^2 倍，这里为 2^2 = 4倍
        # self.conv = nn.Conv2d(in_channels=1, out_channels=4 * (upscale_factor ** 2),
        #                       kernel_size=3, stride=1, padding=1)
        # PixelShuffle层：重排特征图的通道以形成高分辨率的图像
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x
#==================================================

## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################

## Global features Gated-CNN (GCNN)
class FeedForward_out(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_out, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        # self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=7,
        #                         stride=1, padding=7 // 2, groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=7,
                                stride=1, padding=7//2, groups=hidden_features//4, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x = x1 * F.gelu(x2) #+ F.gelu(x1)*x2  #dual???  mish gelu
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x

## Detail feature Gated Feed-Forward Network (DsGFF)
class FeedForward_Duals(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Duals, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_5 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
                               groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
                                        groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1#mish
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
#############################################

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)#Attention
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_out(dim, ffn_expansion_factor, bias)#FeedForward_Dual

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
#########################################################################

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        #assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, out_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad= False)

        self.wt_function = partial(wavelet.wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(wavelet.inverse_wavelet_transform, filters=self.iwt_filter)


    def forward(self, x):

        curr_x = self.wt_function(x)
        curr_x_ll = curr_x[:, :, 0, :, :]
        curr_x_h = curr_x[:, :, 1:4, :, :]
        curr_x_lh = curr_x_h[:,:,0,:,:]
        curr_x_hl = curr_x_h[:,:,1,:,:]
        curr_x_hh = curr_x_h[:,:,2,:,:]

        return curr_x_ll,curr_x_lh,curr_x_hl,curr_x_hh

#================================================================================================
#==============learning upsample ICCV
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='pl', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

#########################################################################

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, input1, input2):
        b, c, h, w = input1.shape

        qkv_1 = self.qkv_dwconv(self.qkv(input1))
        q_1, k_1, v_1 = qkv_1.chunk(3, dim=1)

        q_1 = rearrange(q_1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_1 = rearrange(k_1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_1 = rearrange(v_1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q_1 = torch.nn.functional.normalize(q_1, dim=-1)
        k_1 = torch.nn.functional.normalize(k_1, dim=-1)

        qkv_2 = self.qkv_dwconv(self.qkv(input2))
        q_2, k_2, v_2 = qkv_2.chunk(3, dim=1)

        q_2 = rearrange(q_2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_2 = rearrange(k_2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_2 = rearrange(v_2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q_2 = torch.nn.functional.normalize(q_2, dim=-1)
        k_2 = torch.nn.functional.normalize(k_2, dim=-1)

        cosattn12 = (q_2 @ k_1.transpose(-2, -1)) * self.temperature
        cosattn12 = cosattn12.softmax(dim=-1)
        out12 = (cosattn12 @ v_1)+q_2

        cosattn21 = (q_1 @ k_2.transpose(-2, -1)) * self.temperature
        cosattn21 = cosattn21.softmax(dim=-1)
        out21 = (cosattn21 @ v_2)+q_1

        # out = out12+out21
        #
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w',
        #                 head=self.num_heads, h=h, w=w)

        out12 = rearrange(out12, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out21 = rearrange(out21, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out12 = self.project_out(out12)
        out21 = self.project_out(out21)

        # out = out12*input1 + out21*input2
        out = out12 + out21 +input1 +input2

        #out = self.project_out(out)
        return out


class FeatureInteractionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2., bias=False, LayerNorm_type='WithBias',init_fusion=False):
        super(FeatureInteractionBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attention = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        if init_fusion:
            #self.ffn = Mlp(in_features=dim,ffn_expansion_factor=1, )
            #self.ffn = MLP(dim=dim,mlp_ratio=4,)
            #self.ffn = MlpHead(dim=dim,mlp_ratio=4,)
            #self.ffn = FeedForward_Duals(dim, ffn_expansion_factor, bias)
            self.ffn = FeedForward_out(dim, ffn_expansion_factor, bias)
        else:
            #self.ffn = FeedForward_Dual(dim, ffn_expansion_factor, bias)
            self.ffn = FeedForward_Duals(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):

        x = self.attention(self.norm1(x), self.norm1(y))#####
        x = x + self.ffn(self.norm2(x))

        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class LocalFeatureExtraction(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=2,  #3
                 ):
        super(LocalFeatureExtraction, self).__init__()
        self.Extraction = nn.Sequential(*[DEBlockTrain(default_conv,dim=dim,kernel_size=3) for i in range(num_blocks)])
    def forward(self, x):
        return self.Extraction(x)

class Restormer_Encoder(nn.Module):

    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,#64
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # self.encoder_level1 = nn.Sequential(
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0]//2)]
                                            ,*[ConvFormer(dim=dim) for i in range(num_blocks[0]//2)])

        # self.encoder_level1 = nn.Sequential(
        #     *[Conv2Former(dim=dim) for i in range(num_blocks[0] // 2)],
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0] // 2)])

        # self.encoder_level1 = nn.Sequential(
        #     *[ConvFormer(dim=dim) for i in range(num_blocks[0] // 2)],
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0] // 2)])


        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        #self.baseFeature = BaseFeatureExtraction_Pool(dim=dim)
        self.FreFeature = WTConv2d(in_channels=dim, out_channels=dim)
        self.detailFeature = DetailFeatureExtraction()
        #self.localFeature = LocalFeatureExtraction()
        self.upsample1 = nn.Sequential(
           nn.ConvTranspose2d(int(dim*3), int(dim), kernel_size=2, stride=2),
           nn.LeakyReLU(),
        )
        self.init_fusion_b = FeatureInteractionBlock(dim=64, num_heads=8, init_fusion=True)
        self.init_fusion_d = FeatureInteractionBlock(dim=64, num_heads=8, init_fusion=False)


    def forward(self, inp_img):
        if inp_img.shape[2] % 2 == 0 and inp_img.shape[3] % 2 ==0:
            inp_img = inp_img
        elif inp_img.shape[2] % 2 != 0 and inp_img.shape[3] % 2 ==0:
            inp_img = inp_img[:,:,:inp_img.shape[2] - 1,:]
        elif inp_img.shape[2] % 2 == 0 and inp_img.shape[3] % 2 !=0:
            inp_img = inp_img[:,:,:,:inp_img.shape[3] - 1]
        else:
            inp_img = inp_img[:,:,:inp_img.shape[2] - 1,:inp_img.shape[3] - 1]
        inp_enc_level1 = self.patch_embed(inp_img)
        fre_ll, fre_lh, fre_hl, fre_hh = self.FreFeature(inp_enc_level1)
        fre_lo = F.interpolate(fre_ll, size=inp_enc_level1.shape[-2:], mode='bilinear')
        fre_hi = self.upsample1(torch.cat((fre_lh, fre_hl, fre_hh), dim=1))
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        #detail_feature = self.localFeature(out_enc_level1)
        base_feature = self.init_fusion_b(base_feature,fre_lo)
        detail_feature = self.init_fusion_d(detail_feature, fre_hi)

        return base_feature, detail_feature#self.detailFeature(detail_feature)


class Restormer_Decoder_Phase(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,#64
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',

                 ):

        super(Restormer_Decoder_Phase, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        # self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
        #                                     bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0]//2)]
                                            ,*[ConvFormer(dim=dim) for i in range(num_blocks[0]//2)])

        # self.encoder_level2 = nn.Sequential(
        #     *[Conv2Former(dim=dim) for i in range(num_blocks[0] // 2)],
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0] // 2)])

        # self.encoder_level2 = nn.Sequential(
        #     *[ConvFormer(dim=dim) for i in range(num_blocks[0] // 2)],
        #     *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                        bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0] // 2)])



        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            #nn.GELU(),
            #nn.SiLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()
        self.fusion1 = CGAFusion(dim = dim)
        self.fusion2 = GLFusion(dim=dim)

    def forward(self, inp_img, feature_ll, feature_detail):

        if inp_img.shape[2] % 2 == 0 and inp_img.shape[3] % 2 ==0:
            inp_img = inp_img
        elif inp_img.shape[2] % 2 != 0 and inp_img.shape[3] % 2 ==0:
            inp_img = inp_img[:,:,:inp_img.shape[2] - 1,:]
        elif inp_img.shape[2] % 2 == 0 and inp_img.shape[3] % 2 !=0:
            inp_img = inp_img[:,:,:,:inp_img.shape[3] - 1]
        else:
            inp_img = inp_img[:,:,:inp_img.shape[2] - 1,:inp_img.shape[3] - 1]
        out_enc_level0 = self.fusion1(feature_ll, feature_detail)
        #out_enc_level0 = self.reduce_channel(out_enc_level0)
        #out_enc_level0 = self.fusion2(feature_ll, feature_detail)
        out_enc_level1 = self.encoder_level2(out_enc_level0)+out_enc_level0
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)

        return self.sigmoid(out_enc_level1), out_enc_level0

if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD_1 = Restormer_Decoder_Phase().cuda()
