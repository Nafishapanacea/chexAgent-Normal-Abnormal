import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import XRayDataset
from transforms import get_train_transform, get_val_transform
from model import get_model
from utils import train_one_epoch, validate


def train_model():

    train_csv = "/home/jupyter-nafisha/X-ray/CSVs/train_combined.csv"
    val_csv = "/home/jupyter-nafisha/X-ray/CSVs/valid_combined.csv"
    img_dir = '/home/common'

    # Datasets
    train_dataset = XRayDataset(train_csv, img_dir, transform=get_train_transform())
    val_dataset = XRayDataset(val_csv, img_dir, transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    EPOCHS = 100
    best_val_acc = 0.0  # to store best accuracy

    for epoch in range(EPOCHS):
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

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
    torch.save(model.state_dict(), "last_model.pth")
    print("Last model saved as last_model.pth")


if __name__ == "__main__":
    train_model()

