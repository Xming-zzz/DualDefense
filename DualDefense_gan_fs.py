import torch
import torch.nn as nn
from adv import Adversary_Init
from encoder_fs import ResNetUNet
from decoder_fs import Decoder
from faceswap_pytorch.models import Autoencoder


class DualDefense(nn.Module):
    def __init__(self, message_size, in_channels, device):
        super().__init__()
        self.encoder = ResNetUNet(message_size)
        self.df_model = Autoencoder()
        self.decoder = Decoder(message_size)
        self.adv_model = Adversary_Init(in_channels)
        checkpoint = torch.load('/f/0626/faceswap_pytorch/save/ckpt/LFW-George_W_Bush-Colin_Powell-0126.t7')
        # checkpoint = torch.load('/f/0626/faceswap_pytorch/save/ckpt/LFW-Tony_Blair-Donald_Rumsfeld-0126.t7')
        # checkpoint = torch.load('/f/0626/faceswap_pytorch/save/ckpt/521.t7')
        # checkpoint = torch.load('/f/0626/faceswap_pytorch/save/ckpt-casia/70314.t7')
        self.df_model.load_state_dict(checkpoint['state'])

        if device:
            self.encoder = self.encoder.to(device)
            self.df_model = self.df_model.to(device)
            self.decoder = self.decoder.to(device)
            self.adv_model = self.adv_model.to(device)

    def encode(self, x, message):
        return self.encoder(x, message)

    def deepfake1(self, x, type):
        return self.df_model(x, type)

    def adv(self, x):
        return self.adv_model(x)

    def decode(self, x):
        return self.decoder(x)
