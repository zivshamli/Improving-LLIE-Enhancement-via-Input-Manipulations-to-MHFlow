import torch
import torch.nn as nn
from time import time
from models.modules.ORB import ORB
from models.modules.CBAM import CBAM
from models.modules import thops
from models.modules.URestormer import URestormerBlock
from models.modules.UORB import UORB
from models.MHFlow_model import calculate_batch_edge_map

class TransConvBlock(nn.Module):
    def __init__(self, dim, with_em=False):
        super(TransConvBlock, self).__init__()
        self.TransBlock = URestormerBlock(dim=dim)
        self.ConvBlock = UORB(dim=dim)
        self.Global_Modulator = nn.Sequential(CBAM(gate_channels=dim, with_em=with_em), nn.Sigmoid())
        self.Local_Modulator = nn.Sequential(CBAM(gate_channels=dim, with_em=with_em), nn.Sigmoid())
        self.with_em=with_em

    def forward(self, x, y, em: torch.Tensor = None):
        if self.with_em:
            assert em != None, "EXPECTED Edge Map" # em = calculate_edge_map(x)

        x = self.TransBlock(x)
        y = self.ConvBlock(y)

        scale_x = self.Global_Modulator((y, em))
        x = x * scale_x

        scale_y = self.Local_Modulator((x, em))
        y = y * scale_y

        return x, y


class TransConvNet(nn.Module):
    def __init__(self, dim, n_block):
        super(TransConvNet, self).__init__()
        self.blocks = nn.ModuleList([TransConvBlock(dim=dim, with_em=True) for i in range(n_block)])
        # self.blocks = nn.ModuleList([TransConvBlock(dim=dim, with_em=(i==0)) for i in range(n_block)])

    def forward(self, x, y, em: torch.Tensor=None):
        
        for i,block in enumerate(self.blocks):
            # if i == 0:
            #   x,y = block(x,y, em) 
            # else:
            x, y = block(x, y, em)
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
