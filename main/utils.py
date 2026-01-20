import torch

def encode_view(orientation):
    # 0: PA, 1: AP, 2: Lateral
    if orientation == "lateral":
        return 2
    if orientation == "PA":
        return 0
    return 1

def encode_sex(sex):
    return 0 if sex == "M" else 1


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for image, view, sex, label in train_loader:
        image = image.to(device)
        view = view.to(device)
        sex = sex.to(device)
        label = label.float().unsqueeze(1).to(device)  # [B,1]

        optimizer.zero_grad()

        outputs = model(image, view, sex)              # [B,1] logits
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == label).sum().item()
        total_samples += label.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, val_loader, criterion, DEVICE):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():

        for image, view, sex, label in val_loader:
            image = image.to(DEVICE)
            view = view.to(DEVICE)
            sex = sex.to(DEVICE)
            label = label.float().unsqueeze(1).to(DEVICE)
            
            outputs = model(image, view, sex)   
            loss = criterion(outputs, label)

            total_loss += loss.item()* label.size(0)

            # Accuracy
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)

    return total_loss / total_samples, total_correct / total_samples