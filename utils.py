from tqdm import tqdm
import torch
import os
from PIL import Image
import sys


def train(encoder, decoder, optimizer, loss_fun, loader, ep, device, history, args):
    encoder.train()
    decoder.train()

    # allocation
    total_loss = 0
    n_total_entries = 0
    intermed_loss = 0
    n_intermed_entries = 0

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
        loss_vall_cpu = loss_val.detach().cpu().numpy()
        total_loss += loss_vall_cpu
        n_total_entries += bs
        intermed_loss += loss_vall_cpu
        n_intermed_entries += bs

        # update and save history
        if n_total_entries / args.batch_size % args.save_hist_every == 0:
            history = history.append({"epoch": ep, "n_updates": ep * len(loader.dataset) + n_total_entries,
                                      "train_loss": intermed_loss/ n_intermed_entries}, ignore_index=True)
            history.to_csv(os.path.join(args.folder, 'history.csv'), index=False)
            # reset variables
            intermed_loss = 0
            n_intermed_entries = 0

        # save model
        if n_total_entries / args.batch_size % args.save_model_every == 0:
            save_model(ep, encoder, decoder, optimizer, args.folder)

        # Update train bar
        progress_bar.desc = process_desc.format(ep, loss_val / bs)
        progress_bar.update(bs)
    progress_bar.close()

    return encoder, decoder, optimizer, history


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

        # delete folder
        # os.rmdir(path_images)


def save_model(ep, encoder, decoder, optimizer, folder):
    # save the model
    torch.save({'epoch': ep,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict()
                },
               os.path.join(folder, 'model.pth'))
    tqdm.write("Save model!")
