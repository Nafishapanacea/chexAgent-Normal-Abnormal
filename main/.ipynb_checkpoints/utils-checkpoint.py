import torch

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)   # shape: [B,1]

        optimizer.zero_grad()

        outputs = model(images)                           # shape: [B,1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- Compute accuracy for binary classification ----
        preds = torch.sigmoid(outputs)                    # convert logits → probabilities
        preds = (preds > 0.5).float()                     # threshold

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # print(images.shape, labels.shape, outputs.shape)

    return total_loss / len(train_loader), total_correct / total_samples


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Accuracy
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(val_loader), total_correct / total_samples
