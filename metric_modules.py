from cat_mod import SEP
from residual.loader import ImageLabelDataset

import numpy as np
import torch
from scipy.stats import pearsonr

def train_classifier(config, peck_obs_num = 1000, n_iter = 10000, encoded_data=None, labels=None, logger = None):
    shuffle_mask = np.arange(len(labels))
    np.random.shuffle(shuffle_mask)

    sep = SEP.SEP(config['num_exemp'], output_space_shape=2, input_space_shape = 64,
    lr = config['lr'], dr=config['dr'], omega = config['omega'], delta = config['delta']) # make kernel gaussian
    results = np.zeros(peck_obs_num)
    peck = True
    counter = 0

    assert n_iter < len(labels), "Number of iterations should not surpass number of datapoints!"
    for i in range(n_iter):
        obs, label = encoded_data[shuffle_mask[i]].reshape(1,-1), labels[shuffle_mask[i]].reshape(1,-1)
        peck = sep.predict(obs)
        
        if peck[0]==1 and not (logger is None):
            logger.log(
                {"peck_edible": label.item()},
                step = counter
                )
            results[counter] = label.item()
            counter += 1
            if counter == peck_obs_num: print(f"Trial finished!"); break
        if peck:
            sep.fit(obs, label)
        elif np.random.random() < 0.01:
            sep.fit(obs, label)


def encode_dataset(encoder, dataloader, device='cpu'):
    """
    Encodes images and labels using the provided encoder (e.g., a CNN or transformer).
    
    Args:
        encoder: encoder function(!) not an encoder itself
        dataloader: PyTorch DataLoader yielding (image_tensor, label) pairs.
        device: 'cpu' or 'cuda'
        
    Returns:
        Tuple of (encoded_features_numpy, labels_numpy)
    """
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to('cpu').numpy()

            # Forward pass through the encoder
            features = encoder(images)  # output shape: (batch_size, feature_dim)
            # Move features to CPU and convert to NumPy
            features = features.detach().cpu().numpy()

            all_features.append(features)
            all_labels.append(labels)

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    return features_np, labels_np

def compare_embeddings(emb1, emb2):
    dissim1 = 1. - np.corrcoef(emb1)
    dissim2 = 1. - np.corrcoef(emb2)

    triu_indices = np.triu_indices_from(dissim1, k=1)
    flat1 = dissim1[triu_indices]
    flat2 = dissim2[triu_indices]
    
    # Compute second-order similarity (Pearson correlation)
    r, _ = pearsonr(flat1, flat2)
    return r
