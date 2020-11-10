import nltk
from collections import Counter
from pycocotools.coco import COCO
from tqdm import tqdm


class Vocabulary(object):
    def __init__(self, config):
        self.path_captions = config.path_captions
        self.threshold = config.vocab_threshold

        # vocabulary (string to index; index to string lists)
        self.str2idx = []
        self.idx2str = []

        # tokenizer
        self.tokenizer = lambda s: nltk.tokenize.wordpunct_tokenize(s.lower())

        # special tokens
        self.PAD = '<pad>'
        self.START = '<start>'
        self.END = '<end>'
        self.UNK = '<unk>'

    def build_vocab(self):
        """
        builds the vocabulary from the captions if the word in the caption occurs more often than 'threshold' times.
        """

        # setup the data
        coco = COCO(self.path_captions)
        counter = Counter()
        ids = coco.anns.keys()

        # tokenize the data and count the occurrences of tokens
        process_desc = "Tokenize captions"
        progress_bar = tqdm(initial=0, leave=True, total=len(ids), desc=process_desc, position=0)
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = self.tokenizer(caption)
            counter.update(tokens)
            progress_bar.update(1)
        progress_bar.close()

        # Sort all tokens by count
        token_count = sorted(((cnt, token) for token, cnt in counter.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the dummy tokens,
        # and then all tokens, sorted by frequency.
        self.idx2str = [self.PAD, self.START, self.END, self.UNK] + \
                       [token for cnt, token in token_count if cnt >= self.threshold]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.str2idx = {token: i for i, token in enumerate(self.idx2str)}

    def encode(self, token_list):
        """
        encodes a list of tokens to their idx, adds a start and end token idx.
        """
        unk_idx = self.str2idx[self.UNK]
        encoded = [self.str2idx.get(token, unk_idx) for token in token_list]
        return [self.str2idx[self.START]] + encoded + [self.str2idx[self.END]]

    def decode(self, idx_list):
        """
        decodes a list of indices to tokens. Removes start and end token.
        """
        decoded = [self.idx2str[idx] for idx in idx_list]
        return decoded[1:-1]

    def __len__(self):
        return len(self.str2idx)
