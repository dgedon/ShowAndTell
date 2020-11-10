import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os


class MocoDataSet(Dataset):
    def __init__(self, _path_images, _path_captions, _vocab, _transform):
        self.path_images = _path_images
        self.coco = COCO(_path_captions)
        self.vocab = _vocab
        self.transform = _transform

        self.ids = list(self.coco.anns.keys())

    def __getitem__(self, idx):
        """
        input: index
        output: torch tensors of image and a list of index encoded caption
        """
        # get annotation id
        annotation_id = self.ids[idx]

        # get caption
        caption = self.coco.anns[annotation_id]['caption']
        # convert caption (tokenize, encoder to idx)
        tokens = self.vocab.tokenizer(caption)
        caption_idx = self.vocab.encode(tokens)

        # get images
        img_id = self.coco.anns[annotation_id]['image_id']
        img_name = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.path_images, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(img)

        return caption_idx, image

    def __len__(self):
        return len(self.ids)


class ImageCaptionBatcher:
    def __init__(self, vocab):
        self.pad_idx = vocab.str2idx[vocab.PAD]

    def __call__(self, data):
        # unpack data
        caps, imgs = zip(*data)

        # merge images in tensor
        images = torch.stack(imgs, 0)

        # merge captions in tensor (pad for longest caption)
        # how long is the longest caption
        cap_lengths = [len(caption) for caption in caps]
        max_length = max(cap_lengths)
        # build the caption tensor. Pad the shorter captions
        captions = torch.as_tensor([caption + [self.pad_idx] * (max_length - len(caption)) for caption in caps])

        return images, captions  # , cap_lengths


def get_dataloaders(config, vocab, mean, std, shuffle=True):
    batch_size = config.batch_size
    path_images = config.path_images_preprocessed
    path_captions = config.path_captions

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load the dataset
    mscoco_dataset = MocoDataSet(path_images, path_captions, vocab, transform)

    # load the batcher
    batcher = ImageCaptionBatcher(vocab)

    # Data loader for MS COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=mscoco_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=batcher)
    return data_loader
