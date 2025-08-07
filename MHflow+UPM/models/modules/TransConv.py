import torch
import torch.nn as nn
from time import time
from models.modules.ORB import ORB
from models.modules.CBAM import CBAM
from models.modules import thops
from models.modules.URestormer import URestormerBlock
from models.modules.UORB import UORB
from models.MHFlow_model import calculate_upm

class TransConvBlock(nn.Module):
    def __init__(self, dim, with_upm=False):
        super(TransConvBlock, self).__init__()
        self.TransBlock = URestormerBlock(dim=dim)
        self.ConvBlock = UORB(dim=dim)
        self.Global_Modulator = nn.Sequential(CBAM(gate_channels=dim, with_upm=with_upm), nn.Sigmoid())
        self.Local_Modulator = nn.Sequential(CBAM(gate_channels=dim, with_upm=with_upm), nn.Sigmoid())
        self.with_upm=with_upm

    def forward(self, x, y, upm: torch.Tensor = None):
        if self.with_upm:
            assert upm != None, "EXPECTED UPM" # upm = calculate_upm(x)
        
        x = self.TransBlock(x)
        y = self.ConvBlock(y)

        scale_x = self.Global_Modulator((y, upm))
        x = x * scale_x

        scale_y = self.Local_Modulator((x, upm))
        y = y * scale_y

        return x, y


class TransConvNet(nn.Module):
    def __init__(self, dim, n_block):
        super(TransConvNet, self).__init__()
        self.blocks = nn.ModuleList([TransConvBlock(dim=dim, with_upm=True) for i in range(n_block)])
        # self.blocks = nn.ModuleList([TransConvBlock(dim=dim, with_upm=(i==0)) for i in range(n_block)])

    def forward(self, x, y, upm: torch.Tensor=None):

        for i,block in enumerate(self.blocks):
            # if i == 0:
            #   x,y = block(x,y, upm) 
            # else:
            x, y = block(x, y, upm)
        return x, y


if __name__ == "__main__":
    # model = TransConvBlock(dim=96, depth=2)
    model = TransConvNet(dim=96, n_block=3)
    print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
    begin = time()
    x = torch.ones(1, 96, 200, 300)
    y = torch.ones(1, 96, 200, 300)
    x, y = model(x, y)
    end = time()
    print(end - begin)
