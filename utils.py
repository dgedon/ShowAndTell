from tqdm import tqdm
import torch
import os
from PIL import Image
import numpy as np
import sys
from torch.nn.utils.rnn import pack_padded_sequence


def train(encoder, decoder, optimizer, loss_fun, loader, ep, device, history, args, do_train=True):
    if do_train:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    # allocation
    total_loss = 0
    n_total_updates = 0
    intermed_loss = 0
    n_intermed_updates = 0

    # initialise progress bar
    process_desc = "Epoch {:2d}: Loss: {:2.3e}; Perplexity: {:.4f}"
    progress_bar = tqdm(initial=0, leave=True, total=len(loader),
                        desc=process_desc.format(ep, 0, 0), position=0)
    for (images, captions, lengths) in loader:
        # to device
        images = images.to(device)
        captions = captions.to(device)

        if do_train:
            # forward pass over model
            vis_features = encoder(images)
            output = decoder(vis_features, captions, lengths)

            # get targets (packed because of padding of the captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # compute loss function
            loss_val = loss_fun(output, targets)

            # zero out the gradients
            decoder.zero_grad()
            encoder.zero_grad()

            # Backward pass
            loss_val.backward()
            # Optimize
            optimizer.step()
        else:
            with torch.no_grad():
                # forward pass over model
                vis_features = encoder(images)
                output = decoder(vis_features, captions, lengths)

                # get targets (packed because of padding of the captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # compute loss function
                loss_val = loss_fun(output, targets)

        # Update
        loss_val_cpu = loss_val.item()
        total_loss += loss_val_cpu
        n_total_updates += 1
        intermed_loss += loss_val_cpu
        n_intermed_updates += 1

        # update history
        if n_total_updates % args.save_hist_every == 0:
            history = history.append({"epoch": ep, "n_updates": ep * len(loader) + n_total_updates,
                                      "loss": intermed_loss / n_intermed_updates,
                                      "perplexity": np.exp(intermed_loss / n_intermed_updates)}, ignore_index=True)
            history.to_csv(os.path.join(args.folder, 'history.csv'), index=False)
            # reset variables
            intermed_loss = 0
            n_intermed_updates = 0

        # save model
        if n_total_updates % args.save_model_every == 0:
            save_model(ep, encoder, decoder, optimizer, args.folder)

        # Update train bar
        progress_bar.desc = process_desc.format(ep, loss_val_cpu, np.exp(loss_val_cpu))
        progress_bar.update(1)
    progress_bar.close()

    return intermed_loss / n_intermed_updates, encoder, decoder, optimizer, history


def preprocess_images(path_images, path_output, img_resize, dataset):
    """
    only do preprocessing if not yet done before.
    Preprocessing is considered as done before if output path exists and is not empty
    """

    try:
        # if these operations run, then the directory exists
        list_output_path = os.listdir(path_output)
        test = list_output_path[0]
        tqdm.write("\tPreprocessing {} images already done.".format(dataset))
    except:
        # generate output path
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # get all images in the folder
        image_name_list = os.listdir(path_images)

        # go through image list and resize
        process_desc = "Resize images"
        progress_bar = tqdm(initial=0, leave=True, total=len(image_name_list), desc=process_desc, position=0)
        for image_name in image_name_list:
            with Image.open(os.path.join(path_images, image_name)) as img:
                format = img.format
                # resize
                img = img.resize([img_resize, img_resize], Image.ANTIALIAS)
                # save to new path
                img.save(os.path.join(path_output, image_name), format)
                progress_bar.update(1)
        progress_bar.close()


def save_model(ep, encoder, decoder, optimizer, folder):
    # save the model
    torch.save({'epoch': ep,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict()
                },
               os.path.join(folder, 'model.pth'))
