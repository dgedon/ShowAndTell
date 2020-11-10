from tqdm import tqdm
import torch
import os
from PIL import Image
import sys


def train(encoder, decoder, optimizer, loss_fun, loader, ep, device):
    encoder.train()
    decoder.train()

    # allocation
    total_loss = 0
    n_entries = 0

    # initialise progress bar
    process_desc = "Epoch {:2d}: train - Loss: {:2.3e}"
    progress_bar = tqdm(initial=0, leave=True, total=len(loader.dataset),
                        desc=process_desc.format(ep, 0), position=0)
    for (images, captions) in loader:
        # current batch_size
        bs = images.size(0)
        # to device
        images = images.to(device)
        captions = captions.to(device)

        # forward pass over model
        vis_features = encoder(images)
        output = decoder(vis_features, captions)

        # compute loss function
        loss_val = loss_fun(output, captions)

        # zero out the gradients
        decoder.zero_grad()
        encoder.zero_grad()

        # Backward pass
        loss_val.backward()
        # Optimize
        optimizer.step()

        # Update
        total_loss += loss_val.detach().cpu().numpy()
        n_entries += bs

        # Update train bar
        progress_bar.desc = process_desc.format(ep, loss_val/n_entries)
        progress_bar.update(bs)
    progress_bar.close()

    return total_loss / n_entries, encoder, decoder


def preprocess_images(config):
    """
    only do preprocessing if not yet done before.
    Preprocessing is considered as done before if output path exists and is not empty
    """
    path_images = config.path_images
    img_resize = config.img_resize
    path_output = config.path_images_preprocessed

    try:
        # if these operations run, then the directory exists
        list_output_path = os.listdir(path_output)
        test = list_output_path[0]
        tqdm.write("\tPreprocessing already done.")

    except:

        # generate output path
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        # get all images in the folder
        image_name_list = os.listdir(path_images)

        # go through image list and resize
        process_desc = "\tResize images"
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

        # delete folder
        # os.rmdir(path_images)
