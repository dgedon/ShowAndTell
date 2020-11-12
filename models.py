import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


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
        # max number of words in sequence for sampling
        self.max_words = config.max_words

        # model parameter
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.vocab_size = vocab_size

        # word embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.lin_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, vis_feat, captions, lengths):
        """
        sizes:
        visual features = batch_size x embed_dim
        captions = batch_size x max_length
        length = list of length max_length
        """
        # get embeddings of captions
        emb = self.embedding(captions)
        # combine visual features with embeddings
        inp = torch.cat([vis_feat.unsqueeze(1), emb], 1)
        # pack in packed sequence because of padding (otherwise we just learn padding)
        inp_packed = pack_padded_sequence(inp, lengths, batch_first=True)
        # rnn forward pass. Hidden states are initialised with zero since it is not differently defined
        rnn_out1, _ = self.rnn(inp_packed)
        # take the data out of the packed sequence
        rnn_out2 = rnn_out1.data
        # last linear layer
        output = self.lin_layer(rnn_out2)

        return output

    def sample(self, vis_feat, hidden_rnn=None):
        # allocate sample ids
        sample_ids = []
        # first input to rnn are the visual features
        inp = vis_feat.unsqueeze(0)
        # for at least self.max_words steps
        for _ in range(self.max_words):
            # rnn outputs
            out_rnn, hidden_rnn = self.rnn(inp, hidden_rnn)
            # linear layer
            out_lin = self.lin_layer(out_rnn)
            # get the highest scoring index
            sample_ids.append(out_lin.argmax())
            # get embedding of this sample as new input
            inp = self.embedding(sample_ids[-1]).view(1, 1, -1)
        # generate list of samples
        sample = [id.item() for id in sample_ids]

        return sample
