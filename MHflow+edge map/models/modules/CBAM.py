import torch
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
import cv2


def calculate_edge_map(image):
    """
    Calculate an edge map using Sobel and Laplacian filters.

    Parameters:
        image (np.ndarray): Input BGR image (as read by cv2.imread)

    Returns:
        edge_map (np.ndarray): Combined edge map.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))

    # Sobel X and Y
    sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    # Combine Sobel and Laplacian
    edge_map = cv2.bitwise_or(sobelCombined, lap)

    return edge_map



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
    def __init__(self, with_em=False, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.with_em = with_em
        # Input channels: avg+max (2) plus em channel if enabled
        in_channels = 2 + (1 if self.with_em else 0)
        self.spatial = BasicConv(in_channels, 1, kernel_size,
                                 stride=1,
                                 padding=(kernel_size - 1) // 2,
                                 relu=False)

    def forward(self, x, em: torch.Tensor = None):
        """Spatial attention, optionally integrating the em passed from outside."""
        # avg+max pooling descriptors
        x_compress = self.compress(x)  # [B,2,H,W]

        if self.with_em:
            if em is None:
                raise ValueError("with_em=True but no em provided to SpatialGate.")
            # Resize em to match descriptor spatial dims
            if em.shape[2:] != x_compress.shape[2:]:
                em = F.interpolate(em, size=x_compress.shape[2:],
                                    mode='bilinear', align_corners=False)
            sa_input = torch.cat([x_compress, em], dim=1)  # [B,3,H,W]
        else:
            sa_input = x_compress  # [B,2,H,W]

        att = self.spatial(sa_input)     # [B,1,H,W]
        scale = torch.sigmoid(att)       # [B,1,H,W]
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False, with_em=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(with_em=with_em)
        self.with_em = with_em

    def forward(self, x):
        # If x is a tuple, unpack
        if isinstance(x, (list, tuple)):
            x, em = x
        else:
            em = None

        if em is None:
            em = torch.zeros_like(x[:, :1])  # Fallback 1-channel dummy em

        elif em.dim() == 3:
            em = em.unsqueeze(1)

        if em.shape[-2:] != x.shape[-2:]:
            em = F.interpolate(em, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out, em=em)
        return x_out + x


if __name__ == "__main__":
    # model = MPRNet()
    model = CBAM(gate_channels=16)
    em = calculate_edge_map(torch.randn(1, 3, 336, 336))
    x = torch.randn(1, 16, 84,  84)
    y = model(x, em)
    print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
