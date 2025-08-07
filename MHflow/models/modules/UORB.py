import torch
import torch.nn as nn
from models.modules.ORB import ORB
from models.modules.module_util import Upsample, Downsample


class UORB(nn.Module):
    def __init__(self, dim):
        super(UORB, self).__init__()
        # Encoder
        self.encoder_level1 = ORB(n_feat=dim, kernel_size=3, reduction=4, act=nn.PReLU(), bias=True, num_cab=1)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = ORB(n_feat=dim * 2 ** 1, kernel_size=3, reduction=4, act=nn.PReLU(), bias=True, num_cab=1)
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = ORB(n_feat=dim * 2 ** 2, kernel_size=3, reduction=4, act=nn.PReLU(), bias=True, num_cab=1)

        # Decoder
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)
        self.decoder_level2 = ORB(n_feat=dim * 2 ** 1, kernel_size=3, reduction=4, act=nn.PReLU(), bias=True, num_cab=1)
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=False)
        self.decoder_level1 = ORB(n_feat=dim, kernel_size=3, reduction=4, act=nn.PReLU(), bias=True, num_cab=1)

    def forward(self, x):
        inp_enc_level1 = x
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        x = x + out_dec_level1

        return x


if __name__ == "__main__":
    model = UORB(dim=64)
    # model = TransformerBlock(dim=192, num_heads=8)
    x = torch.randn(1, 64, 200, 300)
    y = model(x)
    print("Parameters of full network %.4f" % (sum([m.numel() for m in model.parameters()]) / 1e6))
