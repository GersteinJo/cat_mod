from runner import Runner
from cat_mod.models.representations.DIM import Encoder
from cat_mod.utils.image_dataset import ImageLabelDataset
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder().to(device)
    root = Path(r'path/to/DIM/weights/')
    enc_file = root / Path('cifar_pretrained150.wgt')
    encoder.load_state_dict(torch.load(str(enc_file), map_location = device))

    pd.read_csv('image_info.csv')['edible'].astype(int).to_csv("test_is_edible.csv")
    dataset = ImageLabelDataset(
        csv_file="/path/to/test_is_edible.csv",
        img_dir="/path/to/image/dir",
        transform=transforms.ToTensor()  # Add any other transforms
    )
    loader = DataLoader(dataset, batch_size=512)
    loader_original = DataLoader(dataset, batch_size=512)

    #encoder_func SPECIFICALLY returns only encoded elements!
    encoder_func = lambda x: encoder(x)[0]

    runner = Runner(
        encoder_func,
        loader,
        loader_original,
        "/path/to/example_config.yaml"
    )

    embeddings, labels, original_images = runner.run("test_rerunning", device, n_iter = 30)
    ################################################################################################ 64 -- dimentionality of embedding space
    df = pd.DataFrame(np.hstack([embeddings, original_images]), columns = [f"emb_{i}" for i in range(64)]+[f"or_{i}" for i in range(32*32*3)])
    df['label'] = labels
    df.to_csv('/path/to/DIM_embedding.csv')
