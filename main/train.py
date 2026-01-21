import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import CXRMulitmodalDataset
from model import CheXagentSigLIPBinary
from utils import train_one_epoch, validate
from transforms import train_transforms
from transformers import AutoModel, AutoProcessor, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
vision_full = AutoModel.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
).to(device, dtype)
vision_encoder = vision_full.vision_model
del vision_full

def train_model():

    img_dir = '/home/common/data_v3'
    train_csv = '/home/jupyter-nafisha/X-ray-covariates/CSVs/train.csv'
    val_csv = "/home/jupyter-nafisha/X-ray-covariates/CSVs/val.csv"
    
    # Datasets
    train_dataset = CXRMulitmodalDataset(train_csv, img_dir, transform=train_transforms)
    val_dataset = CXRMulitmodalDataset(val_csv, img_dir, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Model
    model = CheXagentSigLIPBinary(vision_encoder= vision_encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss & Optimizer
    pos_weight = torch.tensor([2.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.vision_encoder.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4
    )


    # Training
    EPOCHS = 20
    best_val_acc = 0.0  # to store best accuracy

    for epoch in range(EPOCHS):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print("-" * 50)

        # ---- SAVE BEST MODEL ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model updated with val_acc = {best_val_acc:.4f}")
        # break

    # ---- SAVE LAST MODEL ----
    torch.save(model.state_dict(), "last_model.pt")
    print("Last model saved as last_model.pt")


if __name__ == "__main__":
    train_model()

