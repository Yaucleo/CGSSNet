# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
# from SSA import MultiHead_SelfAttention
# import boundary_loss
from GCN import Graph_Conv_Network
device = torch.device("cuda")
# 是否使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING = 1

class DoubleConv(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3, 3, 3), padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)  # 卷积   卷积之后大小通道不变   [4, 96, 64, 64]
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C) [4, 96, 64, 64])
        x = self.norm(x)  # Layernorm
        x = self.pwconv1(x)  # 1*1卷积   [4, 64, 64, 384]   384 = dim(96) * 4    (把(4,64,64)看做整体输入线性层)
        x = self.act(x) # GELU
        x = self.grn(x)
        x = self.pwconv2(x)  # 1*1卷积    [4, 64, 64, 384]
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)  [4, 96, 64, 64]

        x = input + self.drop_path(x)  # 残差




        return x

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
#
#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x



class ConvNeXt(nn.Module):

    def __init__(self, in_chans=1, depths=[3, 3, 9, 3], dims=[16, 32, 64, 128],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        # stem层为transformer中的，4*4卷积，stride=4 替换resnet的池化
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        ###stage2-stage4的3个downsample
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),

            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        ##这里才用到block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]


        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.up1 = nn.ConvTranspose3d(dims[3], dims[2], 2, stride=2)


        self.up2 = nn.ConvTranspose3d(dims[2], dims[1], 2, stride=2)


        self.up3 = nn.ConvTranspose3d(dims[1], dims[0], 2, stride=2)


        self.up4 = nn.ConvTranspose3d(dims[0], 1, 4, stride=4)

        self.conv1 = DoubleConv(dims[3], dims[2])

        self.conv2 = DoubleConv(dims[2], dims[1])

        self.conv3 = DoubleConv(dims[1], dims[0])

        # self.conv1 = DoubleConv(8 * size, 16 * size)


        self.GCN1 = Graph_Conv_Network(in_channels=dims[0], hidden_channels=dims[0] * 2, out_channels=dims[0]).to(device)

        self.GCN2 = Graph_Conv_Network(in_channels=dims[1], hidden_channels=dims[1] * 2, out_channels=dims[1]).to(device)

        self.GCN3 = Graph_Conv_Network(in_channels=dims[2], hidden_channels=dims[2] * 2, out_channels=dims[2]).to(device)

        self.GCN4 = Graph_Conv_Network(in_channels=dims[3], hidden_channels=dims[3] * 2, out_channels=dims[3]).to(device)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')





    def forward_features(self, x):

        ##分类的区别，分类在卷积层输出后拉平(N, C, H, W) -> (N, C)
        # 而分割直接接卷积的输出，里面的结构模块都是一样的
        input = x
        outs = []
        # 测试 去掉stem
        # a = self.downsample_layers
        # x =  self.stem(x)
        # # if i in self.out_indices:
        # norm_layer = getattr(self, f'norm{0}')
        # x_out = norm_layer(x)
        # outs.append(x_out)
        #

        for i in range(4):
            x = self.downsample_layers[i](
                x)  # [4,3,256,256]-->[4, 96, 64, 64]--->[4,192,32,32]--->[4, 384, 16, 16]--.[4,768,8,8]
            x = self.stages[i](x)  # 为什么不加上这一部分
            # [0,1,2,3]
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # 四个特征图
        # x0 = x[0]
        # x1 = x[1]
        # x2 = x[2]
        # x3 = x[3]
        x0 = self.GCN1(x[0])
        x1 = self.GCN2(x[1])
        x2 = self.GCN3(x[2])
        x3 = self.GCN4(x[3])

        # 跳跃连接

        # out1 = self.up1(x3)
        # skip_1 = torch.cat([out1, x2], dim=1)
        # out1 = self.conv1(skip_1)
        # out2 = self.up2(out1)
        # skip_2 = torch.cat([out2, x1], dim=1)
        # out2 = self.conv2(skip_2)
        # out3 = self.up3(out2)
        # skip_3 = torch.cat([out3, x0], dim=1)
        # out = self.conv3(skip_3)
        #
        # out = self.up4(out)
        #
        # out = nn.Sigmoid()(out)




        # 多深度
        out1 = self.up1(x3)
        out1 = self.up2(out1)
        out1 = self.up3(out1)


        # out1 = self.up4(out1)
        #
        out2 = self.up2(x2)
        out2 = self.up3(out2)

        # out2 = self.up4(out2)
        #
        out3 = self.up3(x1)
        out = out1 + out2 + out3 + x0
        out = self.up4(out)

        out = nn.Sigmoid()(out)
        return out


class LayerNorm(nn.Module):
    # 看数据输入是[n,c,w,d]还是[n,w,d,c]来决定参数channels_last or channels_first
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


if __name__ == '__main__':
    # CUDA_LAUNCH_BLOCKING = 1
    print

    # data = torch.randn((1, 1, 288, 288, 32))
    # model = ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128])
    # with torch.no_grad():
    #     out = model(data)
    # print(out)
    # data = torch.randn(1,1,64,64,64)
    # block = Block(1)
    # out = block(data)
    # print(out)