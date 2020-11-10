import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embed_dim = config.embed_dim

        # download pretrained model
        resnet = getattr(models, config.encoder_model.lower())(pretrained=True)
        # replace last (linear) layer
        modules = list(resnet.children())[:-1]

        # save modules of encoder
        self.resnet = nn.Sequential(*modules)
        self.lin_layer = nn.Linear(resnet.fc.in_features, self.embed_dim)
        self.batch_norm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, src_img):
        """
        input: source images
        output: feature vector of source image
        """
        # batch size
        bs = src_img.size(0)

        # do not use gradients for the resnet
        with torch.no_grad():
            src1 = self.resnet(src_img)
        src2 = src1.reshape(bs, -1)
        # linear layer and batch normalization
        src3 = self.lin_layer(src2)
        out = self.batch_norm(src3)

        return out


class Decoder(nn.Module):
    def __init__(self, config, vocab_size):
        super(Decoder, self).__init__()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.max_words = config.max_words
        self.vocab_size = vocab_size

        # word embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.lin_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, vis_feat, captions, lengths):
        """
        sizes:
        visual features =
        captions =
        length =
        """
        emb = self.embedding(captions)
        #TODO
        out = 1

        return out
