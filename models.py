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

    def forward(self, vis_feat, captions):
        """
        sizes:
        visual features = batch_size x embed_dim
        captions = batch_size x max_length
        length = list of length max_length
        """
        # batch size
        bs = captions.size(0)

        # get embeddings of captions
        emb1 = self.embedding(captions)
        emb2 = emb1.permute(1, 0, 2)        # length first
        # combine visual features with embeddings
        inp = torch.cat([vis_feat.unsqueeze(0), emb2], 0)
        # rnn forward pass. Hidden states are initialised with zero since it is not differently defined
        rnn_out1, _ = self.rnn(inp)
        # do not take first output (this only takes in the visual features)
        rnn_out2 = rnn_out1[1:]
        # last linear layer
        out1 = self.lin_layer(rnn_out2)
        # transpose output to be of shape (batch_size, vocab_size, length)
        out2 = out1.permute(1, 2, 0)

        return out2
