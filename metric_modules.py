from cat_mod import SEP
from residual.loader import ImageLabelDataset

import numpy as np
import torch
from scipy.stats import pearsonr

def train_classifier(config, peck_obs_num = 200, n_iter = 10000, encoded_data=None, labels=None, logger = None):
    shuffle_mask = np.arange(len(labels))
    np.random.shuffle(shuffle_mask)

    sep = SEP.SEP(config['num_exemp'], output_space_shape=2, input_space_shape = 64,
    lr = config['lr'], dr=config['dr'], omega = config['omega'], delta = config['delta'])
    results = np.zeros(peck_obs_num)
    peck = [1]
    counter = 0

    assert n_iter < len(labels), "Number of iterations should not surpass number of datapoints!"
    for i in range(n_iter):
        
        obs, label = encoded_data[shuffle_mask[i]].reshape(1,-1), labels[shuffle_mask[i]:shuffle_mask[i]+1]
        # print(label)
        peck = sep.predict(obs)

        if peck[0]==1:# and not (logger is None):
            # print("HEHEHEHEHEHHHHEHHEHEHEHEHEHEH")
            if logger:
                logger.log(
                    {"peck_edible": label[0]},
                    step = counter
                    )
            results[counter] = label.item()
            counter += 1
            if counter == peck_obs_num: print(f"Trial {i} finished!"); break
            sep.fit(obs, label)
        elif np.random.random() < 0.05:
            # print(f"label {label}")
            sep.fit(obs, label)


def encode_dataset(encoder, dataloader, orig_loader, device='cpu'):
    """
    Encodes images and labels using the provided encoder (e.g., a CNN or transformer).
    
    Args:
        encoder: encoder function(!) not an encoder itself
        dataloader: PyTorch DataLoader yielding (image_tensor, label) pairs.
        device: 'cpu' or 'cuda'
        
    Returns:
        Tuple of numpy arrays (encoded_features_numpy, labels_numpy, original_images_numpy)
    """
    all_embeddings = []
    all_labels = []
    all_original_images = []

    with torch.no_grad():
        for (images_dino, targets), (images_orig, _) in zip(loader, original_loader):
            images_dino = images_dino.to(device, non_blocking=True)
            embeddings = encoder_function(images_dino).cpu()
            all_embeddings.append(embeddings)
            all_labels.append(targets)
            all_original_images.append(images_orig)

    embeddings = torch.cat(all_embeddings)  # [N, D]
    labels = torch.cat(all_labels)
    original_images = torch.cat(all_original_images)
    original_images = original_images.reshape(original_images.shape[0], -1)

    return (embeddings.numpy(),
            labels.numpy(),
            original_images.numpy())

    

def compare_embeddings(emb1, emb2):
    dissim1 = 1. - np.corrcoef(emb1)
    dissim2 = 1. - np.corrcoef(emb2)

    triu_indices = np.triu_indices_from(dissim1, k=1)
    flat1 = dissim1[triu_indices]
    flat2 = dissim2[triu_indices]

    # Compute second-order similarity (Pearson correlation)
    r, _ = pearsonr(flat1, flat2)
    return r
