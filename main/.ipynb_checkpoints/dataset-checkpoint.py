import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class XRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.df['label_num'] = self.df['label'].map({"Normal": 0, "Abnormal": 1})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["image_id"]
        # if not image_id.endswith(".png"):
        #     image_id += ".png"

        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label_num"]).long()

        return image, label
