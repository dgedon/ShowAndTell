from tqdm import tqdm
import torch


def train(encoder, decoder, optimizer, loss_fun, loader, ep, device):
    encoder.train()
    decoder.train()

    # allocation
    total_loss = 0
    n_entries = 0

    # initialise progress bar
    process_desc = "Epoch {:2d}: train - Loss: {:2.3e}"
    progress_bar = tqdm(initial=0, leave=True, total=loader.dataset.data.size(0),
                        desc=process_desc.format(ep, 0), position=0)
    for (images, captions, lengths) in loader:
        # current batch_size
        bs = images.size(0)
        # to device
        images = images.to(device)
        captions = captions.to(device)

        # forward pass over model
        vis_features = encoder(images)
        output = decoder(vis_features, captions, lengths)

        # compute loss function
        loss_val = loss_fun(output, targets)

        # zero out the gradients
        decoder.zero_grad()
        encoder.zero_grad()

        # Backward pass
        loss_val.backward()
        # Optimize
        optimizer.step()



        # TODO: from tutorial
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()


        if do_train:
            optimizer.zero_grad()
            model.zero_grad()


            # compute loss function
            loss_val = loss_fun()

            # Backward pass
            loss_val.backward()
            # Optimize
            optimizer.step()
        else:
            with torch.no_grad():


                # compute loss function
                loss_val = loss_fun()
        # Update
        total_loss += loss_val.detach().cpu().numpy()
        n_entries += bs

        # Update train bar
        progress_bar.desc = process_desc.format(ep, total_loss / n_entries)
        progress_bar.update(bs)
    progress_bar.close()

    return total_loss / n_entries, encoder, decoder