import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


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
        self.dropout_val = config.dropout

        # decoder layers
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_val)
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
        emb1 = self.embedding(captions)
        # dropout on embeddings
        emb2 = self.dropout(emb1)
        # combine visual features with embeddings
        inp = torch.cat([vis_feat.unsqueeze(1), emb2], 1)
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

    def beam_search_sample(self, vis_feat, beam_size, hidden_rnn=None):
        # define Softmax
        m = nn.Softmax(dim=2)
        # first input to rnn are the visual features
        inp = vis_feat.unsqueeze(0)
        # allocate sample beams; log prob, current sentence, current layer, next input
        sample_beams = [[0.0, [], hidden_rnn, inp]]
        # for at most self.max_words steps
        for _ in range(self.max_words):
            # allocate possible candidates
            all_candidates = []
            for sample in sample_beams:
                # rnn outputs
                out_rnn, new_hidden_rnn = self.rnn(sample[3], sample[2])
                # linear layer
                out_lin_1 = self.lin_layer(out_rnn)
                # Softmax the previous layer
                out_lin = m(out_lin_1)
                # Find the values and indices of k largest values
                k_vals, k_inds = torch.topk(out_lin, beam_size)
                k_vals, k_inds = k_vals.detach().numpy(), k_inds.detach().numpy()
                for i in range(beam_size):
                    # next input
                    inp = self.embedding(torch.tensor(k_inds[0][0][i])).view(1, 1, -1)
                    # Create new candidates; new sentence, new log prob, new layer, next input
                    candidate = [sample[0] - np.log(k_vals[0][0][i]), sample[1] + [k_inds[0][0][i]], new_hidden_rnn,
                                 inp]
                    all_candidates.append(candidate)
            # Sort the candidates
            all_candidates.sort(key=lambda x: x[0])
            # Take the top k
            sample_beams = all_candidates[:beam_size]
        # Take the best candidate
        sample = sample_beams[0][1]
        return sample
