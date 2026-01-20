import torch.nn as nn
import torchvision.models as models
import torch

def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device= "cpu"
    
    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    
    return model.to(device)