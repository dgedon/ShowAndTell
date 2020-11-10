import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from dataloader import get_dataloaders
from utils import train, preprocess_images, save_model
from vocabulary import Vocabulary
from models import Encoder, Decoder

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(add_help=False)
    # learning parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    # vocabulary parameters
    parser.add_argument('--path_captions', default=os.getcwd() + '/data/annotations/captions_train2014.json',
                        help='path to captions (annotations) of images')
    parser.add_argument('--vocab_threshold', type=int, default=5,
                        help='minimum number of word occurrences (default: 5)')
    # image parameters
    parser.add_argument('--path_images', default=os.getcwd() + '/data/train2014/',
                        help='path to training images')
    parser.add_argument('--path_images_preprocessed', default=os.getcwd() + '/data/preprocessed_train2014/',
                        help='path to training images')
    parser.add_argument('--img_resize', type=int, default=224,
                        help='resize images to size x size (default: 224)')
    # Model parameters
    parser.add_argument('--encoder_model', choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        default="resnet50",
                        help='name of pretrained encoder model (default: renset50)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='dimension of word embeddings (default: 512)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='dimension of hidden rnn states (default: 512)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of rnn layers (default: 1)')
    # inference parameter
    parser.add_argument('--max_words', type=int, default=20,
                        help='maximal number of words per caption (default: 20)')
    parser.add_argument('--path_test_image', default=os.getcwd() + '/test_data/example.png',
                        help='maximal number of words per caption (default: 20)')
    # system settings
    parser.add_argument('--folder', default=os.getcwd() + '/trained_model1',
                        help='storage folder, where the model will be stored')
    parser.add_argument('--save_hist_every', type=int, default=50,
                        help='save the history as average of every n iterations (default: 50)')
    parser.add_argument('--save_model_every', type=int, default=500,
                        help='save the model after every n iterations (default: 500)')
    parser.add_argument('--no_training', type=bool, default=True,
                        help='if set True then no training but only evaluation is done (default: False)')
    args = parser.parse_args()

    tqdm.write("Set folder and save config...")
    # set storage folder
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # save configuration
    with open(os.path.join(args.folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')
    # set seed
    torch.manual_seed(args.seed)
    tqdm.write("Done!")

    tqdm.write("Set device...")
    # set the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write("Using {}!".format(device))

    tqdm.write("Pre-process images...")
    preprocess_images(args)
    tqdm.write("Done!")

    tqdm.write("Construct / Load vocabulary...")
    # construct or load vocabulary
    vocab = Vocabulary(args)
    vocab.build_vocab()
    vocab_size = len(vocab)
    tqdm.write("Done!")

    tqdm.write("Set train/test loader...")
    # channel wise empirical mean and std for normalisation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # get data and set loaders
    train_loader = get_dataloaders(args, vocab, mean, std)
    tqdm.write("Done!")

    tqdm.write("Initialise model...")
    # initialise the model
    encoder = Encoder(args).to(device=device)
    decoder = Decoder(args, vocab_size).to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    # for encoder do not use cnn as parameters, for decoder use all parameters
    parameters = list(encoder.lin_layer.parameters()) + \
                 list(encoder.batch_norm.parameters()) + \
                 list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr)
    tqdm.write("Done!")

    tqdm.write("Define loss criterion...")
    loss_crit = nn.CrossEntropyLoss()
    tqdm.write("Done!")

    tqdm.write("Run training...")
    # initialisation of values

    if not args.no_training:
        history = pd.DataFrame(columns=["epoch", "n_updates", "train_loss"])
        # training can be interrupted by keyboard interrupt
        try:
            for ep in range(args.epochs):
                # run training
                encoder, decoder, optimizer, history = train(encoder, decoder, optimizer, loss_crit, train_loader,
                                                             ep, device, history, args)
                # save the model
                save_model(ep, encoder, decoder, optimizer, args.folder)
            tqdm.write("Training Done!")

        except KeyboardInterrupt:
            tqdm.write("Keyboard interrupt. Stop training!")

    tqdm.write("Create and save loss figure...")
    # open history file
    df = pd.read_csv(os.path.join(args.folder, 'history.csv'))
    # plot and save figure
    plt.figure()
    plt.plot(np.asarray(df.n_updates),np.asarray(df.train_loss), label='train loss')
    plt.title('Training Curve')
    plt.xlabel('updates')
    plt.ylabel('CE Loss')
    plt.legend()
    plt.yscale('log')
    # plt.show()
    plt.savefig(os.path.join(args.folder, 'Training_curve'))
    plt.close()
    tqdm.write("Done!")

    tqdm.write("Test model on image...")
    # load model
    ckpt = torch.load(os.path.join(args.folder, 'model.pth'), map_location=lambda storage, loc: storage)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Prepare an image
    image = Image.open(args.path_test_image)
    image = image.resize([args.img_resize, args.img_resize], Image.ANTIALIAS)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)
    # set model to evaluation
    encoder.eval()
    decoder.eval()
    # Generate an caption from the image
    vis_features = encoder(image_tensor)
    sampled_ids = decoder.sample(vis_features)

    # obtain sentence from word ids
    tokens = vocab.decode(sampled_ids)
    try:
        # find end token
        i = tokens.index(vocab.END)
        # reduce list up to end token
        tokens = tokens[:i]
    except:
        pass
    # combine to sentence
    sentence = ' '.join(tokens)

    # Print out the image and the generated caption
    tqdm.write("Generated image caption is: '{}'".format(sentence))
    image = Image.open(args.path_test_image)
    plt.imshow(np.asarray(image))
