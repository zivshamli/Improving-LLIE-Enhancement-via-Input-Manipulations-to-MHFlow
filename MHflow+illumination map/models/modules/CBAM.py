import torch
from torch import nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, with_illum_map=False, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.with_illum_map = with_illum_map
        # Input channels: avg+max (2) plus illum_map channel if enabled
        in_channels = 2 + (1 if self.with_illum_map else 0)
        self.spatial = BasicConv(in_channels, 1, kernel_size,
                                 stride=1,
                                 padding=(kernel_size - 1) // 2,
                                 relu=False)

    def forward(self, x, illum_map: torch.Tensor = None):
        """Spatial attention, optionally integrating the illum_map passed from outside."""
        # avg+max pooling descriptors
        x_compress = self.compress(x)  # [B,2,H,W]

        if self.with_illum_map:
            if illum_map is None:
                raise ValueError("with_illum_map=True but no illum_map provided to SpatialGate.")
            # Resize illum_map to match descriptor spatial dims
            if illum_map.shape[2:] != x_compress.shape[2:]:
                illum_map = F.interpolate(illum_map, size=x_compress.shape[2:],
                                    mode='bilinear', align_corners=False)
            sa_input = torch.cat([x_compress, illum_map], dim=1)  # [B,3,H,W]
        else:
            sa_input = x_compress  # [B,2,H,W]

        att = self.spatial(sa_input)     # [B,1,H,W]
        scale = torch.sigmoid(att)       # [B,1,H,W]
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False, with_illum_map=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(with_illum_map=with_illum_map)
        self.with_illum_map = with_illum_map

    def forward(self, x):
        # If x is a tuple, unpack
        if isinstance(x, (list, tuple)):
            x, illum_map = x
        else:
            illum_map = None

        if illum_map is None:
            illum_map = torch.zeros_like(x[:, :1])  # Fallback 1-channel dummy illum_map

        elif illum_map.dim() == 3:
            illum_map = illum_map.unsqueeze(1)

        if illum_map.shape[-2:] != x.shape[-2:]:
            illum_map = F.interpolate(illum_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out, illum_map=illum_map)
        return x_out + x


if __name__ == "__main__":
    # model = MPRNet()
    # model = CBAM(gate_channels=16)
    # illum_map = zero_map(torch.randn(1, 3, 336, 336))
    # x = torch.randn(1, 16, 84,  84)
    # y = model(x, illum_map)
    # print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
    pass