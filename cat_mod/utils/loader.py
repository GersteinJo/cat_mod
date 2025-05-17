from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class ImageLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]  # Assuming first column is image names
        img_path = f"{self.img_dir}/{img_name}.png"
        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx, 1]  # Assuming second column is labels

        if self.transform:
            image = self.transform(image)

        return image, label