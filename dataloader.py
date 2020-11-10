import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO



class DataSet(Dataset):
    def __init__(self, path_images, path_captions, vocab, transforms):
        self.path_images = path_images
        self.coco = COCO(path_captions)
        self.vocab = vocab
        self.transforms = transforms

        self.ids = list(self.coco.anns.keys())

    def __getitem__(self, idx):
        # TODO
        #return self.X[idx], self.Y[idx], self.Pos[idx]

    def __len__(self):
        return len(self.ids)

def batcher():
    #TODO


def get_dataloaders(config, vocab, mean, std, shuffle=True):
    batch_size = config.batch_size

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # TODO

    # Data loader for MS COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    #data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=batcher)
