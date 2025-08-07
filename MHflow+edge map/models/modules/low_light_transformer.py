import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.ConditionEncoder import RRDB
from models.modules.TransConv import TransConvNet
from time import time
from timm.models.layers import trunc_normal_


class low_light_transformer(nn.Module):
    def __init__(self, nf=64, n_layer=6, HR_in=False):
        super(low_light_transformer, self).__init__()
        self.nf = nf
        self.HR_in = True if HR_in else False

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        # ConvNextBlock = functools.partial(Block, dim=nf)
        RRDB_Block = functools.partial(RRDB, nf=nf)
        self.feature_extraction = mutil.make_layer(RRDB_Block, 6)
        # self.feature_extraction = mutil.make_layer(ConvNextBlock, n_layer)
        self.TransConv_Hybrid = TransConvNet(dim=nf, n_block=n_layer)

        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, em: torch.Tensor=None):

        x_center = x
        block_results = {}
        block_idxs = {5}

        # Shallow
        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        fea = L1_fea_2
        for idx, m in enumerate(self.feature_extraction.children()):
            fea = m(fea)
            block_results["block_{}".format(idx)] = fea
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        # Deep
        fea_trans = fea
        fea_conv = fea
        fea_trans, fea_conv = self.TransConv_Hybrid(fea_trans, fea_conv, em)
        fea = fea + fea_trans + fea_conv

        out_noise = fea
        fea_down2 = out_noise
        fea_down4 = self.downconv1(
            F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                          recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)
        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))

        results = {
            'fea_up0': fea_down8,
            'fea_up1': fea_down4,
            'fea_up2': fea_down2,
        }
        for k, v in block_results.items():
            results[k] = v

        return results


if __name__ == '__main__':
    # print(torch.__version__)
    x = torch.randn(1, 3, 400, 600)
    model = low_light_transformer(nf=96, n_layer=3, HR_in=True)
    print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
    begin = time()
    y = model(x)
    end = time()
    print(end - begin)
    # print(z)
