import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import encode_view, encode_sex
from transformers import AutoModel, AutoProcessor, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

class CXRMulitmodalDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transforms = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs['pixel_values'].squeeze(0)  

        view = encode_view(row['orientation'])
        sex = encode_sex(row['sex'])
        
        view = torch.tensor(view, dtype=torch.long)
        sex = torch.tensor(sex, dtype=torch.long)
        
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)
        
        return pixel_values, view, sex, label