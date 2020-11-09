import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # TODO

    def forward(self, src_img):
        # TODO
        return src_img


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # TODO

    def forward(self, enc_feat):
        # TODO
        return enc_feat
