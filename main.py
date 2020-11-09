import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils import get_dataloaders, train
from vocabulary import check_and_get_vocab
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
    # input parameters
    parser.add_argument('--rand_crop_size', type=int, default=224,
                        help='randomly crop input images to size x size (default: 224)')
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='dimension of word embeddings (default: 512)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='dimension of hidden rnn states (default: 512)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of rnn layers (default: 1)')
    # system settings
    parser.add_argument('--folder', default=os.getcwd() + '/run1',
                        help='storage folder, where the model will be stored')
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

    tqdm.write("Construct / Load vocabulary...")
    # construct or load vocabulary
    # TODO
    vocab = check_and_get_vocab(args)

    tqdm.write("Set train/test loader...")
    # get data and set loaders
    train_loader, test_loader = get_dataloaders(args)
    tqdm.write("Done!")

    tqdm.write("Initialise model...")
    # initialise the model
    encoder = Encoder(args).to(device=device)
    decoder = Decoder(args).to(device=device)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    # for encoder do not use cnn as parameters, for decoder use all parameters
    parameters = list(encoder.linear.parameters) + list(encoder.bn.parameters) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr)
    tqdm.write("Done!")

    tqdm.write("Define loss criterion...")
    loss_crit = nn.CrossEntropyLoss()
    tqdm.write("Done!")

    tqdm.write("Run training...")
    # initialisation of values

    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss"])
    # training can be interrupted by keyboard interrupt
    try:
        best_val_loss = np.inf
        for ep in range(args.epoch):
            # run training
            train_loss, encoder, decoder = train(encoder, decoder, optimizer, loss_crit, train_loader, args, ep, device,
                                                 do_train=True)
            # run testing
            valid_loss, encoder, decoder = train(encoder, decoder, optimizer, loss_crit, train_loader, args, ep, device,
                                                 do_train=False)

            # Print message
            message = 'Epoch {:2d}: \t\t\tTrain Loss {:2.3e} \t\tValid Loss {:2.3e}'. \
                format(ep, train_loss[-1], valid_loss)
            tqdm.write(message)

            # update and save history
            history = history.append({"epoch": ep, "train_loss": train_loss,
                                      "valid_loss": valid_loss}, ignore_index=True)
            history.to_csv(os.path.join(args.folder, 'history.csv'), index=False)

            # check for best model
            if valid_loss < best_val_loss:
                # update best loss
                best_val_loss = valid_loss
                # save best model
                torch.save({'epoch': ep,
                            'encoder': encoder.state_dict(),
                            'decoder': decoder.state_dict(),
                            'optimizer': optimizer.state_dict()
                            },
                           os.path.join(args.folder, 'model.pth'))
                tqdm.write("Save model!")
        tqdm.write("Training Done!")

    except KeyboardInterrupt:
        tqdm.write("Keyboard interrupt. Stop training!")

    #TODO: testing/sampling
